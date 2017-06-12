<CsoundSynthesizer>
<CsOptions>
</CsOptions>

<CsInstruments>

sr = 44100
nchnls = 2
0dbfs = 1

gitest_down = 1
gitest_up = 2
gitest_both = 3

instr 1
  aout = 0
  if $TEST_TYPE == gitest_down then
    aenv linseg 1000, 1, 10
    aout poscil 1, aenv
  endif

  if $TEST_TYPE == gitest_up then
    aenv linseg 1000, 1, 10000
    aout poscil 1, aenv
  endif

  if $TEST_TYPE == gitest_both then
    aenv_down linseg 1000, 1, 10
    aenv_up linseg 1000, 1, 10000
    aout = poscil(0.5, aenv_down) + poscil(0.5, aenv_up)
  endif

  out aout
endin

</CsInstruments>  

<CsScore>
i1 0 1


</CsScore>
</CsoundSynthesizer>

