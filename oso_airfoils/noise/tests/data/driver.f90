SUBROUTINE XTHICK_CALC (DELTAP,DSTRS,DSTRP)
USE BLParams
IMPLICIT NONE
REAL(8) :: DELTAP,DSTRS,DSTRP
DELTAP = d99(2)
DSTRS = d_star(1)
DSTRP = d_star(2)
RETURN
END SUBROUTINE XTHICK_CALC

PROGRAM DRIVER
USE Third_Octave_Bands
USE BLParams
IMPLICIT NONE
REAL(8) :: FRCEN(NumBands), SPLP(NumBands), SPLS(NumBands)
REAL(8) :: SPLA(NumBands), SPLTBL(NumBands), SPLLAM(NumBands)
REAL(8) :: SPLBLNT(NumBands), SPLTIP(NumBands)
REAL(8) :: C0, VISC, L, R, THETA, PHI, C, U, ALP, H, PSI
REAL(8) :: DELTAP, DSTRS, DSTRP, DBARH, DBARL
INTEGER :: IC, IU, IA, IT, IH, IPS, ITH, IPH, NFREQ
REAL(8) :: CHORDS(3), US(3), ALPHAS(8), HFACS(2), PSIS(3)
REAL(8) :: THETAS(4), PHIS(3), MACHS(2)
LOGICAL :: ROUND

DATA CHORDS /0.05, 0.2286, 1.0/
DATA US /20.0, 63.92, 120.0/
DATA ALPHAS /0.0, 1.0, 3.0, 5.5, 8.0, 12.0, 14.0, 20.0/
DATA HFACS /0.001, 0.01/
DATA PSIS /0.0, 12.5, 14.0/
DATA THETAS /30.0, 90.0, 140.0, 179.0/
DATA PHIS /30.0, 70.0, 90.0/
DATA MACHS /0.1, 0.2/

C0   = 337.7559
VISC = 1.4529E-5
L    = 0.509
R    = 1.22
THETA = 90.0
PHI   = 90.0
NFREQ = NumBands
FRCEN = Third_Octave

OPEN(10, FILE='golden.csv')

! ---- THICK correlations ----
DO IC = 1, 3
  DO IU = 1, 3
    DO IA = 1, 8
      DO IT = 0, 2
        C = CHORDS(IC); U = US(IU); ALP = ALPHAS(IA)
        CALL THICK(C, U, ALP, IT, DELTAP, DSTRS, DSTRP, C0, VISC)
        WRITE(10,'(A,3(F12.6,","),I2,3(",",E16.8))') 'THICK,', C, U, ALP, IT, DELTAP, DSTRS, DSTRP
      END DO
    END DO
  END DO
END DO

! ---- Directivity ----
DO ITH = 1, 4
  DO IPH = 1, 3
    DO IC = 1, 2
      CALL DIRECTH(MACHS(IC), THETAS(ITH), PHIS(IPH), DBARH)
      CALL DIRECTL(MACHS(IC), THETAS(ITH), PHIS(IPH), DBARL)
      WRITE(10,'(A,3(F12.6,","),2(E16.8,:,","))') 'DIRECT,', MACHS(IC), THETAS(ITH), PHIS(IPH), DBARH, DBARL
    END DO
  END DO
END DO

! ---- TBLTE / LBLVS / BLUNT with prescribed BL (X_BLMethod=2) ----
DO IC = 1, 3
  DO IU = 1, 3
    DO IA = 1, 8
      C = CHORDS(IC); U = US(IU); ALP = ALPHAS(IA)
      ! realistic BL values from the untripped correlation
      CALL THICK(C, U, ALP, 0, DELTAP, DSTRS, DSTRP, C0, VISC)
      d99(2) = DELTAP
      d_star(1) = DSTRS
      d_star(2) = DSTRP
      CALL TBLTE(ALP, C, U, FRCEN, 0, SPLP, SPLS, SPLA, SPLTBL, THETA, PHI, L, R, NFREQ, VISC, C0, 2)
      WRITE(10,'(A,3(F12.6,","),34(E16.8,:,","))') 'TBLTE_P,', C, U, ALP, SPLP
      WRITE(10,'(A,3(F12.6,","),34(E16.8,:,","))') 'TBLTE_S,', C, U, ALP, SPLS
      WRITE(10,'(A,3(F12.6,","),34(E16.8,:,","))') 'TBLTE_A,', C, U, ALP, SPLA
      CALL LBLVS(ALP, C, U, FRCEN, SPLLAM, THETA, PHI, L, R, NFREQ, VISC, C0, 2)
      WRITE(10,'(A,3(F12.6,","),34(E16.8,:,","))') 'LBLVS,', C, U, ALP, SPLLAM
      DO IH = 1, 2
        DO IPS = 1, 3
          H = HFACS(IH) * C
          PSI = PSIS(IPS)
          CALL BLUNT(ALP, C, U, FRCEN, 0, SPLBLNT, THETA, PHI, L, R, H, PSI, NFREQ, VISC, C0, 2)
          WRITE(10,'(A,3(F12.6,","),E16.8,",",F12.6,",",34(E16.8,:,","))') 'BLUNT,', C, U, ALP, H, PSI, SPLBLNT
        END DO
      END DO
    END DO
  END DO
END DO

! ---- Off-axis directivity TBLTE spot checks ----
C = 0.2286; U = 63.92; ALP = 3.0
CALL THICK(C, U, ALP, 0, DELTAP, DSTRS, DSTRP, C0, VISC)
d99(2) = DELTAP; d_star(1) = DSTRS; d_star(2) = DSTRP
DO ITH = 1, 4
  DO IPH = 1, 3
    CALL TBLTE(ALP, C, U, FRCEN, 0, SPLP, SPLS, SPLA, SPLTBL, THETAS(ITH), PHIS(IPH), L, R, NFREQ, VISC, C0, 2)
    WRITE(10,'(A,5(F12.6,","),34(E16.8,:,","))') 'TBLTE_DIR,', C, U, ALP, THETAS(ITH), PHIS(IPH), SPLTBL
  END DO
END DO

! ---- Tip noise ----
DO IA = 1, 8
  DO IT = 0, 1
    ROUND = (IT == 0)
    C = 0.5; U = 80.0
    IF (ALPHAS(IA) > 0.0) THEN
      CALL TIPNOIS(ALPHAS(IA), 1.0D0, C, U, FRCEN, SPLTIP, THETA, PHI, R, NFREQ, VISC, C0, ROUND)
      WRITE(10,'(A,3(F12.6,","),I2,34(",",E16.8))') 'TIP,', C, U, ALPHAS(IA), IT, SPLTIP
    END IF
  END DO
END DO

CLOSE(10)
END PROGRAM DRIVER
