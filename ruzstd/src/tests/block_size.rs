//! Blocks that decode past `Block_Maximum_Size` (128 KiB, RFC 8878 3.1.1.2.4) are refused before
//! their output is written, and a Huffman literals stream stops at its stated count.

use alloc::vec;
use alloc::vec::Vec;

use crate::common::MAX_BLOCK_SIZE;
use crate::decoding::errors::{
    DecodeBlockContentError, DecompressBlockError, DecompressLiteralsError, FrameDecoderError,
};
use crate::decoding::{BlockDecodingStrategy, FrameDecoder};

fn decode(frame: &[u8]) -> Result<Vec<u8>, FrameDecoderError> {
    let mut source = frame;
    let mut decoder = FrameDecoder::new();
    decoder.reset(&mut source)?;
    decoder.decode_blocks(&mut source, BlockDecodingStrategy::All)?;
    Ok(decoder.collect().unwrap_or_default())
}

fn block_error(result: Result<Vec<u8>, FrameDecoderError>) -> DecompressBlockError {
    match result {
        Err(FrameDecoderError::FailedToReadBlockBody(
            DecodeBlockContentError::DecompressBlockError(e),
        )) => e,
        other => panic!("expected a block error, got {:?}", other),
    }
}

fn too_large(result: Result<Vec<u8>, FrameDecoderError>) -> u64 {
    match block_error(result) {
        DecompressBlockError::DecompressedSizeTooLarge { at_least } => at_least,
        other => panic!("expected DecompressedSizeTooLarge, got {:?}", other),
    }
}

/// A frame of a `1 << window_log` window, without checksum or content size: an 8-byte raw block
/// for matches to copy from, then the last block, compressed, holding `block`.
fn frame(window_log: u8, block: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0, (window_log - 10) << 3];
    frame.extend_from_slice(&[8 << 3, 0, 0]);
    frame.extend_from_slice(b"12345678");
    let header = 1 | 2 << 1 | (block.len() as u32) << 3;
    frame.extend_from_slice(&header.to_le_bytes()[..3]);
    frame.extend_from_slice(block);
    frame
}

/// A compressed block of no literals and `count` alike sequences in RLE mode: literal length 0,
/// the second repeated offset, and a match length of `131_074 - spare` (match length code 52,
/// whose 16 extra bits are all ones but for `spare`).
fn sequences_block(count: u32, spare: u16) -> Vec<u8> {
    // raw literals of size 0
    let mut block = vec![0];
    match count {
        0..128 => block.push(count as u8),
        128..0x7F00 => block.extend_from_slice(&[(count >> 8) as u8 + 128, count as u8]),
        _ => {
            let rest = count - 0x7F00;
            block.extend_from_slice(&[255, rest as u8, (rest >> 8) as u8]);
        }
    }
    // literal lengths, offsets and match lengths each in RLE mode, then their one symbol
    block.extend_from_slice(&[0b0101_0100, 0, 0, 52]);
    // each sequence reads its 16 extra bits, backwards from the end, after a marker bit
    let extra = u16::MAX - spare;
    for _ in 0..count {
        block.extend_from_slice(&extra.to_le_bytes());
    }
    block.push(1);
    block
}

#[test]
fn matches_past_the_block_maximum_are_refused() {
    // 1000 sequences of 131074 bytes: 131 MB from a 2 KiB frame
    let frame = frame(10, &sequences_block(1000, 0));
    assert_eq!(frame.len(), 2028);
    assert_eq!(too_large(decode(&frame)), 1000 * 131_074);
}

#[test]
fn a_block_of_more_sequences_than_a_u32_sum_holds_does_not_panic() {
    // the lengths of 33000 such sequences overflowed the u32 sum in execute_sequences
    let frame = frame(10, &sequences_block(33_000, 0));
    assert_eq!(too_large(decode(&frame)), 33_000 * 131_074);
}

#[test]
fn more_sequences_than_a_block_has_room_for_are_refused_before_decoding() {
    // every sequence copies at least 3 bytes
    let count = MAX_BLOCK_SIZE / 3 + 1;
    let mut block = vec![0, 255];
    block.extend_from_slice(&((count - 0x7F00) as u16).to_le_bytes());
    block.extend_from_slice(&[0b0101_0100, 0, 0, 0, 1]);
    assert_eq!(too_large(decode(&frame(10, &block))), 3 * u64::from(count));
}

#[test]
fn literals_past_the_block_maximum_are_refused() {
    // RLE literals stated at 2^20 - 1, the most a 3-byte header holds
    let size = (1u32 << 20) - 1;
    let rle = 1 | 0b11 << 2 | (size << 4);
    let mut block = rle.to_le_bytes()[..3].to_vec();
    block.extend_from_slice(&[b'x', 0]);
    assert_eq!(too_large(decode(&frame(10, &block))), u64::from(size));
}

#[test]
fn huffman_literals_stop_at_their_stated_count() {
    // 4 literals stated, then 4 streams of ones under a 1-bit Huffman table: each bit decodes to
    // a literal, nearly a million of them
    const STREAM: usize = 29_998;
    let tree = [0x80, 0x10];
    let compressed = tree.len() + 6 + 4 * STREAM;
    // compressed literals (type 2), 18-bit sizes (format 3): regenerated, then compressed
    let header = 2 | 3 << 2 | 4u64 << 4 | (compressed as u64) << 22;
    let mut block = header.to_le_bytes()[..5].to_vec();
    block.extend_from_slice(&tree);
    for _ in 0..3 {
        block.extend_from_slice(&(STREAM as u16).to_le_bytes());
    }
    block.extend(core::iter::repeat_n(0xFF, 4 * STREAM));
    // no sequences
    block.push(0);
    match block_error(decode(&frame(10, &block))) {
        DecompressBlockError::DecompressLiteralsError(
            DecompressLiteralsError::DecodedLiteralCountMismatch { decoded, expected },
        ) => {
            assert_eq!(expected, 4);
            assert!(decoded <= 5, "decoded {} literals", decoded);
        }
        other => panic!("expected DecodedLiteralCountMismatch, got {:?}", other),
    }
}

#[test]
fn a_block_of_exactly_the_maximum_decodes() {
    // one match of 131074 - 2 bytes
    let decoded = decode(&frame(17, &sequences_block(1, 2))).unwrap();
    assert_eq!(decoded.len(), 8 + MAX_BLOCK_SIZE as usize);
    // the second repeated offset starts as 4: the raw block's last four bytes, over and over
    assert!(decoded[8..]
        .chunks(4)
        .all(|chunk| chunk == &b"5678"[..chunk.len()]));
}
