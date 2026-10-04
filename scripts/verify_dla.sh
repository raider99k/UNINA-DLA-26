#!/bin/bash

# UNINA-DLA Verification Script
# Usage: ./verify_dla.sh <onnx_file>

ONNX_FILE=${1:-"unina_dla_qat.onnx"}
WIDTH=${2:-768}
HEIGHT=${3:-480}
SPARSE_ENABLED=${4:-0}  # Set to 1 for 2:4 sparse models
ENGINE_FILE="${ONNX_FILE%.*}.engine"

# Build sparsity flag if enabled
SPARSITY_FLAG=""
if [ "$SPARSE_ENABLED" = "1" ]; then
    SPARSITY_FLAG="--sparsity=enable"
    echo "Sparsity ENABLED: Compiling with 2:4 structured sparsity support"
fi

echo "Compiling $ONNX_FILE to $ENGINE_FILE for DLA ($WIDTH x $HEIGHT)..."

# Critical Flags for DLA:
# --useDLACore=0: Target DLA Core 0
# --allowGPUFallback: Allow fallback (we want to check if it happens, not crash)
# --inputIOFormats=fp16:dla_hwc4: Avoids GPU reformatting
# --profilingVerbosity=detailed: Show layer placement
# --sparsity=enable: Enable 2:4 structured sparsity (if trained with --enable_sparsity)

# NOTE: For dla_hwc4, TensorRT requires the input to be logically 3 channels (matching ONNX)
# but it will internally pad the stride to 4 for DLA hardware compliance.
trtexec --onnx=$ONNX_FILE \
        --saveEngine=$ENGINE_FILE \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --inputIOFormats=fp16:dla_hwc4 \
        --outputIOFormats=fp16:chw16 \
        --shapes=images:1x3x${HEIGHT}x${WIDTH} \
        --profilingVerbosity=detailed \
        --noDataTransfers \
        --allowGPUFallback \
        --best \
        --memPoolSize=workspace:1024 \
        $SPARSITY_FLAG \
        --exportLayerInfo=layer_info.json > dla_log.txt 2>&1

echo "Compilation finished. Check layer_info.json for detailed placement."

# Robustness Check: Parse layer_info.json if available, otherwise grep log
if [ -f "layer_info.json" ]; then
    # Look for "implementation": "GPU" in the JSON layer info (excluding reformatting layers if they are unavoidable, 
    # but generally we want strict DLA).
    # Simple grep on JSON for "Device": "GPU" or "implementation": "GPU" depending on trtexec version.
    # Standard trtexec JSON usually has "LayerInformation": [ { "Name": "...", "Implementation": "..." } ]
    # We fail if any layer has Implementation: GPU (excluding some known allowable layers if any, but stricter is better).
    FALLBACK_COUNT=$(grep -i '"Implementation":.*"GPU"' layer_info.json | wc -l)
else
    # Fallback to log parsing if JSON not supported/generated
    # Look for "GPU" in the Layer details section, ignoring header/reformatting info potentially.
    # We refine the grep to look for the table entries usually formatted like:
    # "LayerName ... | GPU"
    FALLBACK_COUNT=$(grep "GPU" dla_log.txt | grep "Layer" | wc -l)
fi

if [ $FALLBACK_COUNT -eq 0 ]; then
    echo "SUCCESS: 100% DLA Residency Achieved!"
    # Optional: Check latency
    echo "Latency Benchmark:"
    tail -n 20 dla_log.txt | grep "mean"
    exit 0
else
    echo "FAILURE: $FALLBACK_COUNT layers fell back to GPU. See dla_log.txt or layer_info.json."
    # Print the specific layers if using JSON
    if [ -f "layer_info.json" ]; then
        grep -B 5 -A 2 -i '"Implementation":.*"GPU"' layer_info.json
    fi
    exit 1
fi
