PROGRAM TNODRV
USE Precision
USE Atmosphere
USE TNOConstants
USE AirfoilParams
USE BLParams
USE Third_Octave_Bands
IMPLICIT NONE
REAL (ReKi):: a,b,answer,abserr,resabs,resasc
REAL (ReKi):: band_width,band_ratio,Spectrum,DBARH
REAL (ReKi):: D,R,SPL_val
INTEGER (4):: n_freq
REAL (ReKi), EXTERNAL :: int2
INTEGER :: side

! s809 NAFNoise case conditions, BL from metafoil qfoil
co = 337.7559; nu = 1.4529e-5; rho = 1.225
Mach = 63.92/337.7559
D = 0.509; R = 1.22
! qfoil TE values: suction (1), pressure (2)
d99(1) = 11.06e-3;  Cf(1) = 0.00035;  d_star(1) = 3.839e-3
d99(2) = 7.48e-3;   Cf(2) = 0.00193;  d_star(2) = 1.357e-3

n_freq = NumBands
band_ratio = 2.**(1./3.)
allocate(omega(n_freq))
CALL DIRECTH(Mach,90.0,90.0,DBARH)

open(10,file='tno_golden.csv')
do i_omega = 1,n_freq
   omega(i_omega) = 2.*pi*Third_Octave(i_omega)
   a = 0.
   b = 10*omega(i_omega)/(Mach*co)
   band_width = Third_Octave(i_omega)*(sqrt(band_ratio)-1./sqrt(band_ratio))
   do side = 1, 2
      ISSUCTION = (side == 1)
      CALL qk61(int2,a,b,answer,abserr,resabs,resasc)
      Spectrum = D/(4.*pi*R**2.)*answer
      SPL_val = 10*log10(Spectrum*DBARH/2.e-5/2.e-5) + 10*log10(band_width)
      write(10,'(F10.1,",",I2,",",E16.8)') Third_Octave(i_omega), side, SPL_val
   enddo
enddo
close(10)
END PROGRAM

SUBROUTINE DIRECTH(M,THETA,PHI,DBAR)
REAL M,MC,THETA,PHI,THETAR,PHIR,DBAR,DEGRAD
DEGRAD = .017453
MC = .8 * M
THETAR = THETA * DEGRAD
PHIR = PHI * DEGRAD
DBAR=2.*SIN(THETAR/2.)**2.*SIN(PHIR)**2./((1.+M*COS(THETAR))*(1.+(M-MC)*COS(THETAR))**2.)
END SUBROUTINE
