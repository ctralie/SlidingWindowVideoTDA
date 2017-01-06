SET NTrials=200
SET packet=0

:packetloop

SET trial=0
:trialsloop
del *.txt
WaveletCoder.exe pendulum.cfg Target=375
BitStreamExtractor.exe stream.wfc stream_out.wfc QualityLevel=1 SpatialLevel=3 TemporalLevel=4 ploss=0.%packet% randomizer=0
WaveletCoder.exe decoder.cfg InputFile=stream_out.wfc OutputFile=dec.yuv
SET filename=..\PendulumEncoded\%packet%_%trial%.avi
del %filename%
avconv -s 160x360 -i dec.yuv -vf crop=160:330:0:0 %filename% 
IF %trial%==%NTrials% GOTO finishedtrials
SET /A trial=trial+1
GOTO trialsloop
:finishedtrials

if %packet%==9 GOTO finishedpacket
SET /A packet=packet+1
GOTO packetloop
:finishedpacket

echo FINISHED