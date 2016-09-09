#!/bin/bash
cargo run --example basics "$@"
cargo run --example events "$@"
cargo run --example img_formats "$@"
cargo run --example info_core "$@"
cargo run --example info "$@"
cargo run --example threads "$@"
cargo run --example timed "$@"
cargo run --example trivial "$@"

cd examples/images
cargo run "$@"
cd -

cd examples/images-safe-clamp
cargo run "$@"
cd -
