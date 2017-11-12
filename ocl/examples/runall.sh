#!/bin/bash

VENDOR=""
while getopts v: option
do
 case "${option}"
 in
 v) VENDOR=$OPTARG;;
 esac
done

# Construct feature list.
FEATURES=""
if [ "$VENDOR" = "mesa" ]; then
    FEATURES=$FEATURES" opencl_vendor_mesa "
fi

# Any features enabled?
if [ "$FEATURES" != "" ]; then
    FEATURES="--features "$FEATURES
fi

cargo run --example async_cycles "$@" $FEATURES
cargo run --example async_menagerie "$@" $FEATURES
cargo run --example async_process "$@" $FEATURES
cargo run --example basics "$@" $FEATURES
cargo run --example device_check "$@" $FEATURES
cargo run --example event_callbacks "$@" $FEATURES
cargo run --example img_formats "$@" $FEATURES
cargo run --example info_core "$@" $FEATURES
cargo run --example info "$@" $FEATURES
# cargo run --example map_buffers "$@" $FEATURES
cargo run --example threads "$@" $FEATURES
cargo run --example timed "$@" $FEATURES
cargo run --example trivial "$@" $FEATURES

cd examples
cd images
cargo update
cargo run "$@" $FEATURES
cd ..

cd images_safe_clamp
cargo update
cargo run "$@" $FEATURES
cd ../..
