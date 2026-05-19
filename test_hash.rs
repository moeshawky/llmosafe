#![no_std]
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub fn hash_str_lowercase(s: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for byte in s.bytes() {
        let b = byte.to_ascii_lowercase();
        hash ^= b as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}
