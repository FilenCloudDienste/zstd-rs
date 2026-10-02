//! Blocks that decode past `Block_Maximum_Size` (RFC 8878 3.1.1.2.4: the smaller of the window and
//! 128 KiB) are refused before their output is written, and Huffman literals stop at their stated
//! count.

use alloc::vec;
use alloc::vec::Vec;

use crate::common::MAX_BLOCK_SIZE;
use crate::decoding::errors::{
    DecodeBlockContentError, DecompressBlockError, DecompressLiteralsError, FrameDecoderError,
};
use crate::decoding::{BlockDecodingStrategy, FrameDecoder};

const RAW: u32 = 0;
const COMPRESSED: u32 = 2;

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
        // not the output itself, which is huge if a check regresses
        Ok(decoded) => panic!("expected a block error, decoded {} bytes", decoded.len()),
        Err(e) => panic!("expected a block error, got {:?}", e),
    }
}

/// The size a block was refused at, and the maximum it was held to.
fn too_large(result: Result<Vec<u8>, FrameDecoderError>) -> (u64, u64) {
    match block_error(result) {
        DecompressBlockError::DecompressedSizeTooLarge { at_least, max } => (at_least, max),
        other => panic!("expected DecompressedSizeTooLarge, got {:?}", other),
    }
}

/// A frame of a `1 << window_log` window, without checksum or content size: an 8-byte raw block
/// for matches to copy from, then the last block, of `block_type`, holding `block`.
fn frame(window_log: u8, block_type: u32, block: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0, (window_log - 10) << 3];
    frame.extend_from_slice(&[8 << 3, 0, 0]);
    frame.extend_from_slice(b"12345678");
    let header = 1 | block_type << 1 | (block.len() as u32) << 3;
    frame.extend_from_slice(&header.to_le_bytes()[..3]);
    frame.extend_from_slice(block);
    frame
}

/// A compressed block of no literals and one sequence in RLE mode: literal length 0, the second
/// repeated offset, and a match length of `131_074 - spare` (match length code 52, whose 16 extra
/// bits are all ones but for `spare`).
fn match_block(spare: u16) -> Vec<u8> {
    // raw literals of size 0, then one sequence
    let mut block = vec![0, 1];
    // literal lengths, offsets and match lengths each in RLE mode, then their one symbol
    block.extend_from_slice(&[0b0101_0100, 0, 0, 52]);
    // the 16 extra bits, read backwards from the end after a marker bit
    block.extend_from_slice(&(u16::MAX - spare).to_le_bytes());
    block.push(1);
    block
}

/// A compressed block of `size` RLE literals stated in a 3-byte header, then `rest`.
fn rle_literals_block(size: u32, rest: &[u8]) -> Vec<u8> {
    let header = 1 | 0b11 << 2 | size << 4;
    let mut block = header.to_le_bytes()[..3].to_vec();
    block.push(b'x');
    block.extend_from_slice(rest);
    block
}

/// A compressed block of no sequences whose literals section states 4 literals, but holds one for
/// every bit of its `streams` streams of `stream_len` ones, under a 1-bit Huffman table.
fn huffman_block(streams: usize, stream_len: usize) -> Vec<u8> {
    let tree = [0x80, 0x10];
    let compressed = (tree.len() + 2 * (streams - 1) + streams * stream_len) as u64;
    // compressed literals (type 2), in one stream with 10-bit sizes (format 0) or four with 18-bit
    // sizes (format 3): regenerated, then compressed
    let (format, bits): (u64, usize) = if streams == 1 { (0, 10) } else { (3, 18) };
    let header = 2 | format << 2 | 4 << 4 | compressed << (4 + bits);
    let mut block = header.to_le_bytes()[..(4 + 2 * bits) / 8].to_vec();
    block.extend_from_slice(&tree);
    for _ in 1..streams {
        block.extend_from_slice(&(stream_len as u16).to_le_bytes());
    }
    block.extend(core::iter::repeat_n(0xFF, streams * stream_len));
    // no sequences
    block.push(0);
    block
}

fn assert_stops_at_stated_count(block: &[u8]) {
    match block_error(decode(&frame(18, COMPRESSED, block))) {
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
fn test_match_past_the_block_maximum_rejected() {
    // one match of 131073 bytes in a 256 KiB window
    let frame = frame(18, COMPRESSED, &match_block(1));
    assert_eq!(too_large(decode(&frame)), (131_073, 131_072));
}

#[test]
fn test_sequence_count_past_the_block_maximum_rejected_before_decoding() {
    // every sequence copies at least 3 bytes. Their match lengths (code 52) need 16 extra bits
    // each, which the bitstream does not hold, so decoding them would fail.
    let count = MAX_BLOCK_SIZE / 3 + 1;
    let mut block = vec![0, 255];
    block.extend_from_slice(&((count - 0x7F00) as u16).to_le_bytes());
    block.extend_from_slice(&[0b0101_0100, 0, 0, 52, 1]);
    let frame = frame(18, COMPRESSED, &block);
    assert_eq!(too_large(decode(&frame)), (3 * u64::from(count), 131_072));
}

#[test]
fn test_literals_past_the_block_maximum_rejected() {
    // RLE literals stated at 2^20 - 1, the most a 3-byte header holds. No sequences section
    // follows, so decoding would fail on that if the literals went unchecked.
    let size = (1 << 20) - 1;
    let frame = frame(18, COMPRESSED, &rle_literals_block(size, &[]));
    assert_eq!(too_large(decode(&frame)), (u64::from(size), 131_072));
}

#[test]
fn test_block_maximum_lowered_to_a_smaller_window() {
    // a 1 KiB window holds raw and compressed blocks alike to 1 KiB
    assert_eq!(
        too_large(decode(&frame(10, RAW, &[b'x'; 1025]))),
        (1025, 1024)
    );
    let block = rle_literals_block(1025, &[0]);
    assert_eq!(
        too_large(decode(&frame(10, COMPRESSED, &block))),
        (1025, 1024)
    );

    assert_eq!(
        decode(&frame(10, RAW, &[b'x'; 1024])).unwrap().len(),
        8 + 1024
    );
    let block = rle_literals_block(1024, &[0]);
    assert_eq!(
        decode(&frame(10, COMPRESSED, &block)).unwrap().len(),
        8 + 1024
    );
}

#[test]
fn test_huffman_literals_in_four_streams_stop_at_their_stated_count() {
    // nearly a million literals in the streams
    assert_stops_at_stated_count(&huffman_block(4, 29_998));
}

#[test]
fn test_huffman_literals_in_one_stream_stop_at_their_stated_count() {
    // a 10-bit compressed size leaves room for 1021 bytes of stream, 8168 literals
    assert_stops_at_stated_count(&huffman_block(1, 1_021));
}

#[test]
fn test_block_of_exactly_the_maximum_decodes() {
    // one match of 131072 bytes in a 256 KiB window
    let decoded = decode(&frame(18, COMPRESSED, &match_block(2))).unwrap();
    assert_eq!(decoded.len(), 8 + MAX_BLOCK_SIZE as usize);
    // the second repeated offset starts as 4: the raw block's last four bytes, over and over
    assert!(decoded[8..]
        .chunks(4)
        .all(|chunk| chunk == &b"5678"[..chunk.len()]));
}
