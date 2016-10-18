#!/bin/bash
cargo run --example basics "$@"
cargo run --example events "$@"
cargo run --example img_formats "$@"
cargo run --example info_core "$@"
cargo run --example info "$@"
cargo run --example map_buffers "$@"
cargo run --example threads "$@"
cargo run --example timed "$@"
cargo run --example trivial "$@"

cd examples/images
cargo update
cargo run "$@"
cd -

cd examples/images-safe-clamp
cargo update
cargo run "$@"
cd -
