//=============================================================================
//=========================== UTILITY FUNCTIONS ===============================
//=============================================================================

/// Pads `len` to make it evenly divisible by `incr`.
pub fn padded_len(len: usize, incr: usize) -> usize {
    let len_mod = len % incr;

    if len_mod == 0 {
        len
    } else {
        let pad = incr - len_mod;
        let padded_len = len + pad;
        debug_assert_eq!(padded_len % incr, 0);
        padded_len
    }
}
