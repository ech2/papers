#/bin/bash

csound gen_test_tones.csd --omacro:TEST_TYPE=1 -f --output=audio/test_tones/sweep_down.wav
csound gen_test_tones.csd --omacro:TEST_TYPE=2 -f --output=audio/test_tones/sweep_up.wav
csound gen_test_tones.csd --omacro:TEST_TYPE=3 -f --output=audio/test_tones/sweep_both.wav
