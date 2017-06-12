<CsoundSynthesizer>
<CsOptions>
</CsOptions>

<CsInstruments>

sr = 44100
nchnls = 1
0dbfs = 1

gicf1 = $cf1
gibw1 = $bw1
gicf2 = $cf2
gibw2 = $bw2


instr 1
  aenv_cf linseg gicf1, p3, gicf2
  aenv_bw linseg gibw1, p3, gibw2
  anoise noise 1, 0
  aout butterbp anoise, aenv_cf, aenv_bw
  aout butterbp aout, aenv_cf, aenv_bw
  out aout
endin

</CsInstruments>

<CsScore>
i1 0 1 
</CsScore>
</CsoundSynthesizer>

