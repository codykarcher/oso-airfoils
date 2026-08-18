MODULE Precision
INTEGER, PARAMETER :: ReKi = 8
END MODULE Precision

MODULE Third_Octave_Bands
USE Precision
INTEGER (4),PARAMETER :: NumBands = 34
REAL (ReKi),PARAMETER :: Third_Octave(NumBands) = (/10.,12.5,16.,20.,25.,31.5,40.,50.,63.,80., &
    100.,125.,160.,200.,250.,315.,400.,500.,630.,800., &
    1000.,1250.,1600.,2000.,2500.,3150.,4000.,5000.,6300.,8000., &
    10000.,12500.,16000.,20000./)
END MODULE Third_Octave_Bands

MODULE BLParams
USE Precision
REAL (ReKi) :: d99(2)
REAL (ReKi) :: Cf(2)
REAL (ReKi) :: d_star(2)
END MODULE BLParams
