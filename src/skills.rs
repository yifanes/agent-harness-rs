//! Skill progressive disclosure for the agent loop.
//!
//! A *skill* is a directory holding a `SKILL.md` (with YAML frontmatter
//! carrying `name` + `description`) plus optional `scripts/` and assets.
//! Instead of dumping every `SKILL.md` body into the system prompt, this
//! module renders a compact catalogue — `name` / `description` / `path` per
//! skill — and lets the model pull the full instructions on demand with its
//! existing `read` tool and run any scripts with `bash`. No bespoke
//! "execute skill" tool is involved.
//!
//! # Local vs remote
//!
//! Where the skill files physically live (local FS vs a remote sandbox) is
//! abstracted behind [`SkillSource`]. This crate ships [`LocalSkillSource`]
//! (`std::fs`) as the default; an embedder running tools in a sandbox injects
//! its own [`SkillSource`] (e.g. one backed by a sandbox process client) so
//! the catalogue is read from wherever the agent's other tools operate.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill source error: {0}")]
    Source(String),
}

/// One skill's catalogue entry. `path` is the skill directory as seen by the
/// agent's tools (i.e. inside the sandbox when running remotely); the model
/// reads `{path}/SKILL.md` for the full instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub path: String,
}

/// Abstracts *where* skill files live. Implementors enumerate skill
/// directories and read each `SKILL.md`. The harness only ever talks to this
/// trait — it never knows whether the bytes come from the local disk or a
/// remote sandbox.
#[async_trait]
pub trait SkillSource: Send + Sync {
    /// List skill directories (absolute paths) under the skills root.
    async fn list_skill_dirs(&self) -> Result<Vec<String>, SkillError>;

    /// Read `{dir}/SKILL.md`. `Ok(None)` when the file is absent (the dir is
    /// then skipped, not treated as an error).
    async fn read_skill_md(&self, dir: &str) -> Result<Option<String>, SkillError>;
}

/// Default [`SkillSource`] reading from the local filesystem. Used in local
/// mode and by standalone library users.
pub struct LocalSkillSource {
    root: PathBuf,
}

impl LocalSkillSource {
    /// `root` is the directory that *contains* the per-skill subdirectories
    /// (e.g. `<cwd>/.harness/skills`).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

#[async_trait]
impl SkillSource for LocalSkillSource {
    async fn list_skill_dirs(&self) -> Result<Vec<String>, SkillError> {
        let mut entries = match tokio::fs::read_dir(&self.root).await {
            Ok(e) => e,
            // Missing root ⇒ no skills, not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(SkillError::Source(e.to_string())),
        };
        let mut dirs = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| SkillError::Source(e.to_string()))?
        {
            if entry
                .file_type()
                .await
                .map(|t| t.is_dir())
                .unwrap_or(false)
            {
                dirs.push(entry.path().to_string_lossy().into_owned());
            }
        }
        dirs.sort();
        Ok(dirs)
    }

    async fn read_skill_md(&self, dir: &str) -> Result<Option<String>, SkillError> {
        let path = Path::new(dir).join("SKILL.md");
        match tokio::fs::read_to_string(&path).await {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(SkillError::Source(e.to_string())),
        }
    }
}

/// Parse `name` and `description` out of a `SKILL.md` YAML frontmatter block
/// (`---` … `---`). Only single-line scalar values are supported; surrounding
/// quotes are stripped. Either field may be `None`.
pub fn parse_skill_frontmatter(content: &str) -> (Option<String>, Option<String>) {
    let Some(after_open) = content.strip_prefix("---") else {
        return (None, None);
    };
    let Some(after_open) = after_open.trim_start_matches(' ').strip_prefix('\n') else {
        return (None, None);
    };
    let Some(close_pos) = after_open.find("\n---") else {
        return (None, None);
    };
    let front_matter = &after_open[..close_pos];

    let mut name = None;
    let mut description = None;
    for line in front_matter.lines() {
        if let Some(rest) = line.strip_prefix("name:") {
            name = unquote_nonempty(rest);
        } else if let Some(rest) = line.strip_prefix("description:") {
            description = unquote_nonempty(rest);
        }
    }
    (name, description)
}

fn unquote_nonempty(raw: &str) -> Option<String> {
    let raw = raw.trim();
    let v = raw
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(raw)
        .trim();
    (!v.is_empty()).then(|| v.to_string())
}

fn dir_basename(dir: &str) -> String {
    dir.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(dir)
        .to_string()
}

/// Reads every skill via a [`SkillSource`] and produces [`SkillMetadata`].
/// A directory without a `SKILL.md` is skipped. When the frontmatter omits
/// `name`, the directory basename is used.
#[derive(Default)]
pub struct SkillLoader;

impl SkillLoader {
    pub fn new() -> Self {
        Self
    }

    pub async fn load(&self, src: &dyn SkillSource) -> Result<Vec<SkillMetadata>, SkillError> {
        let mut out = Vec::new();
        for dir in src.list_skill_dirs().await? {
            let Some(content) = src.read_skill_md(&dir).await? else {
                continue;
            };
            let (name, description) = parse_skill_frontmatter(&content);
            out.push(SkillMetadata {
                name: name.unwrap_or_else(|| dir_basename(&dir)),
                description: description.unwrap_or_default(),
                path: dir,
            });
        }
        Ok(out)
    }
}

/// Renders a compact skill catalogue for injection into the system prompt.
#[derive(Default)]
pub struct SkillPromptRenderer;

impl SkillPromptRenderer {
    pub fn new() -> Self {
        Self
    }

    /// `None` when there are no skills (so the caller can omit the section).
    pub fn render(&self, skills: &[SkillMetadata]) -> Option<String> {
        if skills.is_empty() {
            return None;
        }
        let mut s = String::from(
            "You have access to the following skills. When a task matches one, read its \
             SKILL.md at the given path for the full instructions, then follow them (run any \
             bundled scripts with bash). Available skills:",
        );
        for skill in skills {
            s.push_str("\n- ");
            s.push_str(&skill.name);
            if !skill.description.is_empty() {
                s.push_str(": ");
                s.push_str(&skill.description);
            }
            s.push_str(" (path: ");
            s.push_str(&skill.path);
            s.push_str("/SKILL.md)");
        }
        Some(s)
    }
}

/// Orchestrates load + render. The single entry point an embedder calls at
/// turn/boot setup: hand it a [`SkillSource`], get back the system-prompt
/// fragment (or `None`).
#[derive(Default)]
pub struct SkillsManager {
    loader: SkillLoader,
    renderer: SkillPromptRenderer,
}

impl SkillsManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn load_and_render(
        &self,
        src: &dyn SkillSource,
    ) -> Result<Option<String>, SkillError> {
        let skills = self.loader.load(src).await?;
        Ok(self.renderer.render(&skills))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_parses_name_and_description() {
        let md = "---\nname: pdf-tools\ndescription: \"Work with PDFs\"\n---\n# body\n";
        let (name, desc) = parse_skill_frontmatter(md);
        assert_eq!(name.as_deref(), Some("pdf-tools"));
        assert_eq!(desc.as_deref(), Some("Work with PDFs"));
    }

    #[test]
    fn frontmatter_missing_fields_are_none() {
        assert_eq!(parse_skill_frontmatter("no frontmatter"), (None, None));
        let (name, desc) = parse_skill_frontmatter("---\nname: x\n---\n");
        assert_eq!(name.as_deref(), Some("x"));
        assert_eq!(desc, None);
    }

    #[test]
    fn renderer_emits_name_desc_path_and_no_body() {
        let skills = vec![SkillMetadata {
            name: "pdf-tools".into(),
            description: "Work with PDFs".into(),
            path: "/cwd/.harness/skills/pdf-tools".into(),
        }];
        let out = SkillPromptRenderer::new().render(&skills).unwrap();
        assert!(out.contains("pdf-tools"));
        assert!(out.contains("Work with PDFs"));
        assert!(out.contains("/cwd/.harness/skills/pdf-tools/SKILL.md"));
        // Progressive disclosure: the catalogue must not embed body text.
        assert!(!out.contains("# body"));
    }

    #[test]
    fn renderer_empty_is_none() {
        assert!(SkillPromptRenderer::new().render(&[]).is_none());
    }

    /// In-memory `SkillSource` for end-to-end loader/manager tests.
    struct FakeSource(Vec<(String, Option<String>)>);

    #[async_trait]
    impl SkillSource for FakeSource {
        async fn list_skill_dirs(&self) -> Result<Vec<String>, SkillError> {
            Ok(self.0.iter().map(|(d, _)| d.clone()).collect())
        }
        async fn read_skill_md(&self, dir: &str) -> Result<Option<String>, SkillError> {
            Ok(self
                .0
                .iter()
                .find(|(d, _)| d == dir)
                .and_then(|(_, md)| md.clone()))
        }
    }

    #[tokio::test]
    async fn loader_skips_dirs_without_skill_md_and_falls_back_to_basename() {
        let src = FakeSource(vec![
            (
                "/s/alpha".into(),
                Some("---\nname: alpha-skill\ndescription: A\n---\nbody".into()),
            ),
            ("/s/no-md".into(), None),
            // No frontmatter name → basename fallback.
            ("/s/beta".into(), Some("just text, no frontmatter".into())),
        ]);
        let skills = SkillLoader::new().load(&src).await.unwrap();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "alpha-skill");
        assert_eq!(skills[0].description, "A");
        assert_eq!(skills[1].name, "beta");
        assert_eq!(skills[1].description, "");
    }

    #[tokio::test]
    async fn manager_load_and_render_end_to_end() {
        let src = FakeSource(vec![(
            "/s/alpha".into(),
            Some("---\nname: alpha\ndescription: Do alpha\n---\nbody".into()),
        )]);
        let fragment = SkillsManager::new().load_and_render(&src).await.unwrap();
        let fragment = fragment.expect("one skill ⇒ Some");
        assert!(fragment.contains("alpha"));
        assert!(fragment.contains("Do alpha"));
        assert!(fragment.contains("/s/alpha/SKILL.md"));
    }

    #[tokio::test]
    async fn local_source_lists_and_reads(/* uses a temp dir */) {
        let base = std::env::temp_dir().join(format!("harness_skills_test_{}", std::process::id()));
        let skill_dir = base.join("demo");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        tokio::fs::write(skill_dir.join("SKILL.md"), "---\nname: demo\n---\nx")
            .await
            .unwrap();

        let src = LocalSkillSource::new(&base);
        let dirs = src.list_skill_dirs().await.unwrap();
        assert_eq!(dirs.len(), 1);
        assert!(src.read_skill_md(&dirs[0]).await.unwrap().is_some());

        let _ = tokio::fs::remove_dir_all(&base).await;
    }
}
