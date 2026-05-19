pub fn hash_str(s: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

pub fn hash_str_lowercase(s: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for byte in s.bytes() {
        hash ^= byte.to_ascii_lowercase() as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

fn main() {
    assert_eq!(hash_str_lowercase("Hello"), hash_str("hello"));
}
