cargo run --example async_cycles %*
cargo run --example async_menagerie %*
cargo run --example async_process %*
cargo run --example basics %*
cargo run --example device_check %*
cargo run --example event_callbacks %*
cargo run --example img_formats %*
cargo run --example info_core %*
cargo run --example info %*
# cargo run --example map_buffers %*
cargo run --example threads %*
cargo run --example timed %*
cargo run --example trivial %*

cd examples
cd images
cargo update
cargo run %*
cd ..

cd images_safe_clamp
cargo update
cargo run %*
cd ../..
