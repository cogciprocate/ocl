use super::super::ProQue;

#[test]
#[should_panic]
#[allow(unused_variables)]
fn bad_kernel_variable_names() {    
    let kernel = r#"
        kernel void multiply(global float* buffer, float coeff) {

            not_a_variable + im_with_not_a_variable;

            buffer[get_global_id(0)] *= coeff;
        }
    "#;

    let ocl_pq = ProQue::builder().src(kernel).build().unwrap();
}
