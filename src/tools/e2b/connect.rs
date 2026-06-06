//! Connect Protocol wire codec (https://connectrpc.com/docs/protocol/).
//!
//! envd in agentsphere (and stock e2b) sits behind an AWS ELB that doesn't
//! pass `application/grpc` straight through — but it accepts Connect
//! Protocol over HTTP/1.1 / HTTP/2. The official e2b SDKs use
//! `@connectrpc/connect-web` for the same reason.
//!
//! Connect Protocol summary used here:
//!
//! - **Unary**:
//!   - `POST /<package>.<Service>/<Method>`
//!   - `Content-Type: application/proto`
//!   - body = raw protobuf bytes
//!   - 2xx → response body is raw protobuf
//!   - non-2xx → response body is JSON `{"code":..., "message":..., "details":[]}`
//!
//! - **Streaming** (server-stream / client-stream / bidi):
//!   - `POST /<package>.<Service>/<Method>`
//!   - `Content-Type: application/connect+proto`
//!   - body = stream of `Envelope` frames
//!   - last frame has the end-stream flag set; its payload is JSON
//!     metadata with optional `error` field
//!
//! Envelope wire format:
//!   `[flags: u8] [length: u32 big-endian] [payload: length bytes]`
//!
//! flags:
//!   bit 0 (0x01): compression
//!   bit 1 (0x02): end-stream marker
//!
//! We never compress (envd doesn't advertise gzip in our paths) and we
//! treat any non-zero bit 1 as the end-stream marker.

use bytes::{Buf, BytesMut};
use serde::Deserialize;

pub const FLAG_END_STREAM: u8 = 0b0000_0010;

/// Encode a single data envelope (flags=0).
pub fn encode_data_envelope(payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.push(0u8);
    let len = u32::try_from(payload.len()).expect("payload < 4GiB");
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Parsed envelope: either a data frame or the trailing end-of-stream frame
/// (which carries JSON-encoded trailers, possibly an error).
#[derive(Debug)]
pub enum Envelope {
    Data(Vec<u8>),
    EndOfStream(EndOfStream),
}

#[derive(Debug, Default)]
pub struct EndOfStream {
    pub error: Option<ConnectError>,
    /// Free-form metadata returned alongside the trailers JSON.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, thiserror::Error)]
#[error("connect error {code}: {message}")]
pub struct ConnectError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub details: Vec<serde_json::Value>,
}

/// Streaming envelope decoder. Feed it bytes as they arrive on the wire;
/// it emits complete envelopes whenever 5-byte header + payload have
/// accumulated. End-of-stream envelopes terminate the stream — callers
/// should observe the [`Envelope::EndOfStream`] variant and stop reading.
#[derive(Debug, Default)]
pub struct EnvelopeDecoder {
    buf: BytesMut,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("declared envelope length {0} exceeds local hard cap {1}")]
    OversizedFrame(u32, usize),
    #[error("trailers payload is not valid JSON: {0}")]
    BadTrailerJson(#[from] serde_json::Error),
}

impl EnvelopeDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append raw bytes from the wire.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Attempt to peel off the next complete envelope. Returns `Ok(None)`
    /// if more bytes are needed.
    pub fn try_next(&mut self) -> Result<Option<Envelope>, DecodeError> {
        const MAX_FRAME: usize = 16 * 1024 * 1024; // 16 MiB safety cap

        if self.buf.len() < 5 {
            return Ok(None);
        }
        let flags = self.buf[0];
        let len = u32::from_be_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]);
        if (len as usize) > MAX_FRAME {
            return Err(DecodeError::OversizedFrame(len, MAX_FRAME));
        }
        let total = 5 + len as usize;
        if self.buf.len() < total {
            return Ok(None);
        }
        // Consume header.
        self.buf.advance(5);
        let payload = self.buf.split_to(len as usize).to_vec();

        if flags & FLAG_END_STREAM != 0 {
            // Trailers payload is a JSON object; we only care about the
            // optional `error` field; ignore everything else for now.
            let parsed: TrailerJson = if payload.is_empty() {
                TrailerJson::default()
            } else {
                serde_json::from_slice(&payload)?
            };
            Ok(Some(Envelope::EndOfStream(EndOfStream {
                error: parsed.error,
                metadata: parsed.metadata,
            })))
        } else {
            Ok(Some(Envelope::Data(payload)))
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct TrailerJson {
    #[serde(default)]
    error: Option<ConnectError>,
    /// Catch-all for extra keys returned by envd / proxies.
    #[serde(flatten)]
    metadata: serde_json::Map<String, serde_json::Value>,
}

/// Unary error body: same JSON shape as the trailer error.
pub fn parse_unary_error(body: &[u8]) -> Option<ConnectError> {
    serde_json::from_slice::<ConnectError>(body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_data_roundtrip() {
        let payload = b"hello world".to_vec();
        let wire = encode_data_envelope(&payload);
        assert_eq!(wire[0], 0);
        assert_eq!(&wire[1..5], &(payload.len() as u32).to_be_bytes());
        assert_eq!(&wire[5..], &payload[..]);

        let mut dec = EnvelopeDecoder::new();
        dec.push(&wire);
        match dec.try_next().unwrap().unwrap() {
            Envelope::Data(d) => assert_eq!(d, payload),
            other => panic!("expected data, got {other:?}"),
        }
        assert!(matches!(dec.try_next(), Ok(None)));
    }

    #[test]
    fn test_decode_split_across_pushes() {
        let payload = b"split frame".to_vec();
        let wire = encode_data_envelope(&payload);

        let mut dec = EnvelopeDecoder::new();
        dec.push(&wire[..3]); // partial header
        assert!(matches!(dec.try_next(), Ok(None)));
        dec.push(&wire[3..7]); // rest of header + 2 bytes payload
        assert!(matches!(dec.try_next(), Ok(None)));
        dec.push(&wire[7..]); // remainder
        match dec.try_next().unwrap().unwrap() {
            Envelope::Data(d) => assert_eq!(d, payload),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_end_of_stream_with_error() {
        let trailer = br#"{"error":{"code":"unimplemented","message":"nope"}}"#.to_vec();
        let mut wire = Vec::new();
        wire.push(FLAG_END_STREAM);
        wire.extend_from_slice(&(trailer.len() as u32).to_be_bytes());
        wire.extend_from_slice(&trailer);

        let mut dec = EnvelopeDecoder::new();
        dec.push(&wire);
        match dec.try_next().unwrap().unwrap() {
            Envelope::EndOfStream(eos) => {
                let err = eos.error.unwrap();
                assert_eq!(err.code, "unimplemented");
                assert_eq!(err.message, "nope");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_end_of_stream_empty_means_success() {
        let mut wire = Vec::new();
        wire.push(FLAG_END_STREAM);
        wire.extend_from_slice(&0u32.to_be_bytes());

        let mut dec = EnvelopeDecoder::new();
        dec.push(&wire);
        match dec.try_next().unwrap().unwrap() {
            Envelope::EndOfStream(eos) => assert!(eos.error.is_none()),
            _ => panic!(),
        }
    }

    #[test]
    fn test_two_envelopes_in_one_push() {
        let a = encode_data_envelope(b"first");
        let b = encode_data_envelope(b"second");
        let mut wire = a;
        wire.extend_from_slice(&b);

        let mut dec = EnvelopeDecoder::new();
        dec.push(&wire);
        assert!(matches!(dec.try_next().unwrap().unwrap(), Envelope::Data(d) if d == b"first"));
        assert!(matches!(dec.try_next().unwrap().unwrap(), Envelope::Data(d) if d == b"second"));
        assert!(matches!(dec.try_next(), Ok(None)));
    }
}
