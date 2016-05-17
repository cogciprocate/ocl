use core;

static kernel: &'static str = r#"
    __kernel void pyrs_ltp(
                __global uchar const* const axn_states,
                __global uchar const* const cel_states,
                __global uchar const* const cel_tft_best_den_ids,
                __global uchar const* const cel_tft_best_den_states,
                __global uchar const* const den_states,
                __global uchar const* const syn_states,
                __private uint const tfts_per_cel,
                __private uint const dens_per_tft_l2,
                __private uint const syns_per_den_l2,
                __private uint const cels_per_cel_grp,
                __private uint const axn_idz_cel_lyr,
                __private int const learning_rate_l2i,
                __private int const rnd,
                __global uchar* const syn_flag_sets,
                __global uchar* const cel_flag_sets,
                __global int* const aux_ints_0,
                __global int* const aux_ints_1,
                __global char* const syn_strengths)
    {
        uint const cel_grp_id = get_global_id(0);
        uint const cel_grp_count = get_global_size(0);
        uint const cel_count = mul24(cel_grp_count, cels_per_cel_grp);
        uint const cel_idz_cel_grp = mul24(cel_grp_id, cels_per_cel_grp);

         // TODO: (EVALUATE) Make 'cels_per_cel_grp' and 'tfts_per_cel' a constant and unroll loops.
         //    - Will mean making a separate program for each layer of pyramidals.
         //    - Could do more harm than good due to program size bloat.
         //    - Possibly do this for tfts only.
        for (uint cel_id_cel_grp = 0; cel_id_cel_grp < cels_per_cel_grp; cel_id_cel_grp++) {
            uint const cel_idx = cel_idz_cel_grp + cel_id_cel_grp;
            uint const cel_axn_idx = axn_idz_cel_lyr + cel_idx;

            uchar cel_flag_set = cel_flag_sets[cel_idx];

            int const cel_is_concrete = axn_states[cel_axn_idx] != 0;
            int const cel_is_vatic = cel_states[cel_idx] != 0;
            int const cel_prev_concrete = (cel_flag_set & (CEL_PREV_CONCRETE_FLAG)) == (CEL_PREV_CONCRETE_FLAG);
            int const cel_prev_vatic = (cel_flag_set & (CEL_PREV_VATIC_FLAG)) == (CEL_PREV_VATIC_FLAG);
            int const cel_best_in_col = (cel_flag_set & (CEL_BEST_IN_COL_FLAG)) == (CEL_BEST_IN_COL_FLAG);

            for (uint tft_id = 0; tft_id < tfts_per_cel; tft_id++) {
                // uint const cel_tft_idx = mad24(cel_idx, tfts_per_cel, tft_id);
                uint const cel_tft_idx = calc_cel_tft_idx(cel_count, cel_idx, tfts_per_cel, tft_id);
                uint const den_idz_tft = cel_tft_idx << dens_per_tft_l2;

                uchar const den_id_tft_best = cel_tft_best_den_ids[cel_tft_idx];

                uint const syn_idz_tft = den_idz_tft << syns_per_den_l2;
                uint const syn_idz_best_den_tft = (den_idz_tft + den_id_tft_best) << syns_per_den_l2;

                int const tuft_is_active = cel_tft_best_den_states[cel_tft_idx] != 0;

                if (cel_is_concrete) {
                    if (tuft_is_active) {
                        // PREVIOUS (CORRECT) PREDICTION (EVERY PYR IN COL): REINFORCE DEN
                        // ANOMALY (NO PREVIOUS PREDICTION, BEST PYR IN COLUMN ONLY): TRAIN NEW DEN
                        if (cel_prev_vatic | cel_best_in_col) {
                            dst_syns__active__stpot_stdep(syn_states, syn_idz_best_den_tft, syns_per_den_l2, rnd,
                                syn_flag_sets, syn_strengths);
                        }
                    }

                    // TODO: Could be moved into above if block
                    cel_flag_set |= CEL_PREV_CONCRETE_FLAG;

                } else if (cel_prev_concrete) {
                    tft_syns_trm(syn_states, syn_idz_tft, syns_per_den_l2 + dens_per_tft_l2, rnd,
                        learning_rate_l2i, syn_flag_sets, aux_ints_0, syn_strengths);

                    cel_flag_set &= ~CEL_PREV_CONCRETE_FLAG;
                }
            }

            cel_flag_set &= ~CEL_PREV_VATIC_FLAG;
            cel_flag_set |= mul24(cel_is_vatic, CEL_PREV_VATIC_FLAG);

            cel_flag_sets[cel_idx] = cel_flag_set;
        }
    }
"#;


#[test]
fn complex_kernel() {

}
