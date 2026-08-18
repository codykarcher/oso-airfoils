      SUBROUTINE LBLVS(ALPSTAR,C,U ,FRCEN,SPLLAM,THETA,PHI,L,R,
     1                 NFREQ,VISC,C0,X_BLMethod)

      USE Third_Octave_Bands



C                  --------------------------------
C                  ***** VARIABLE DEFINITIONS *****
C                  --------------------------------

C       VARIABLE NAME               DEFINITION                  UNITS
C       -------------               ----------                  -----


C       ALPSTAR             ANGLE OF ATTACK                   DEGREES
C       C                  CHORD LENGTH                       METERS
C       C0                 SPEED OF SOUND                     METERS/SEC
C       D                  REYNOLDS NUMBER RATIO              ---
C       DBARH              HIGH FREQUENCY DIRECTIVITY         ---
C       DELTAP             PRESSURE SIDE BOUNDARY LAYER 
C                            THICKNESS                        METERS
C       DSTRP              PRESSURE SIDE BOUNDARY LAYER
C                            DISPLACEMENT THICKNESS           METERS
C       DSTRS              SUCTION SIDE BOUNDARY LAYER
C                            DISPLACEMENT THICKNESS           METERS
C       E                  STROUHAL NUMBER RATIO              ---
C       FRCEN              1/3 OCTAVE FREQUENCIES             HERTZ
C       G1                 SOUND PRESSURE LEVEL FUNCTION      DB
C       G2                 OVERALL SOUND PRESSURE LEVEL
C                            FUNCTION                         DB
C       G3                 OVERALL SOUND PRESSURE LEVEL
C                            FUNCTION                         DB
C       ITRIP              FLAG TO TRIP BOUNDARY LAYER        ---
C       L                  SPAN                               METERS
C       M                  MACH NUMBER                        ---
C       NFREQ              NUMBER OF FREQUENCIES              ---
C       OASPL              OVERALL SOUND PRESSURE LEVEL       DB
C       PHI                DIRECTIVITY ANGLE                  DEGREES
C       R                  OBSERVER DISTANCE FROM SEGMENT     METERS
C       RC                 REYNOLDS NUMBER BASED ON CHORD     ---
C       RC0                REFERENCE REYNOLDS NUMBER          ---
C       SCALE              GEOMETRIC SCALING TERM
C       SPLLAM             SOUND PRESSURE LEVEL DUE TO
C                            LAMINAR MECHANISM                DB
C       STPRIM             STROUHAL NUMBER BASED ON PRESSURE
C                            SIDE BOUNDARY LAYER THICKNESS    ---
C       ST1PRIM            REFERENCE STROUHAL NUMBER          ---
C       STPKPRM            PEAK STROUHAL NUMBER               ---
C       THETA              DIRECTIVITY ANGLE                  DEGREES
C       U                  FREESTREAM VELOCITY                METERS/SEC
C       VISC               KINEMATIC VISCOSITY                M2/SEC



      DIMENSION STPRIM(NumBands)  ,SPLLAM(NumBands)    ,FRCEN(NumBands)

      REAL      L                   ,M
	INTEGER X_BLMethod
 

C      COMPUTE REYNOLDS NUMBER AND MACH NUMBER
C      ---------------------------------------

      M        = U  / C0
      RC       = U  * C/VISC


C      COMPUTE BOUNDARY LAYER THICKNESSES
C      ----------------------------------
      SELECT CASE (X_BLMethod) 
      CASE (2)
	   CALL XTHICK_CALC(DELTAP,DSTRS,DSTRP)
	CASE DEFAULT 
         CALL THICK(C,U ,ALPSTAR,ITRIP,DELTAP,DSTRS,DSTRP,C0,VISC)
	END SELECT



C      COMPUTE DIRECTIVITY FUNCTION
C      ----------------------------

      CALL DIRECTH(M,THETA,PHI,DBARH)



C      COMPUTE REFERENCE STROUHAL NUMBER
C      ---------------------------------

      IF (RC .LE. 1.3E+05) ST1PRIM = .18
      IF((RC .GT. 1.3E+05).AND.(RC.LE.4.0E+05))ST1PRIM=.001756*RC**.3931
      IF (RC .GT. 4.0E+05) ST1PRIM = .28

      STPKPRM  = 10.**(-.04*ALPSTAR) * ST1PRIM



C      COMPUTE REFERENCE REYNOLDS NUMBER
C      ---------------------------------

      IF (ALPSTAR .LE. 3.0) RC0=10.**(.215*ALPSTAR+4.978)
      IF (ALPSTAR .GT. 3.0) RC0=10.**(.120*ALPSTAR+5.263)




C      COMPUTE PEAK SCALED SPECTRUM LEVEL
C      ----------------------------------

      D   = RC / RC0

      IF (D .LE. .3237) G2=77.852*ALOG10(D)+15.328
      IF ((D .GT. .3237).AND.(D .LE. .5689))
     1  G2 = 65.188*ALOG10(D) + 9.125
      IF ((D .GT. .5689).AND.(D .LE. 1.7579))
     1  G2 = -114.052 * ALOG10(D)**2.
      IF ((D .GT. 1.7579).AND.(D .LE. 3.0889))
     1  G2 = -65.188*ALOG10(D)+9.125
      IF (D .GT. 3.0889) G2 =-77.852*ALOG10(D)+15.328


      G3      = 171.04 - 3.03 * ALPSTAR

      SCALE   = 10. * ALOG10(DELTAP*M**5*DBARH*L/R**2)



C      COMPUTE SCALED SOUND PRESSURE LEVELS FOR EACH STROUHAL NUMBER
C      -------------------------------------------------------------

      DO 100 I=1,NFREQ

         STPRIM(I)  = FRCEN(I) * DELTAP / U 
         
         E          = STPRIM(I) / STPKPRM

         IF (E .LT. .5974) G1=39.8*ALOG10(E)-11.12
         IF ((E .GE. .5974).AND.(E .LE. .8545))
     1     G1 = 98.409 * ALOG10(E) + 2.0
         IF ((E .GE. .8545).AND.(E .LT. 1.17))
     1     G1 = -5.076+SQRT(2.484-506.25*(ALOG10(E))**2.)
         IF ((E .GE. 1.17).AND.(E .LT. 1.674))
     1     G1 = -98.409 * ALOG10(E) + 2.0
         IF (E .GE. 1.674) G1=-39.80*ALOG10(E)-11.12

         SPLLAM(I) = G1 + G2 + G3 + SCALE

  100 CONTINUE

      RETURN
      END
      SUBROUTINE TBLTE(ALPSTAR,C,U ,FRCEN,ITRIP,SPLP,SPLS,
     1              SPLALPH,SPLTBL,THETA,PHI,L,R,NFREQ,VISC,C0,
     2              X_BLMethod)



C                  --------------------------------
C                  ***** VARIABLE DEFINITIONS *****
C                  --------------------------------



C       VARIABLE NAME               DEFINITION                  UNITS
C       -------------               ----------                  -----

C       A                  STROUHAL NUMBER RATIO                 ---
C       A0                 FUNCTION USED IN 'A' CALCULATION      ---
C       A02                FUNCTION USED IN 'A' CALCULATION      ---
C       AA                 'A' SPECTRUM SHAPE EVALUATED AT
C                             STROUHAL NUMBER RATIO              DB
C       ALPSTAR            ANGLE OF ATTACK                     DEGREES
C       AMAXA              MAXIMUM 'A' CURVE EVALUATED AT
C                            STROUHAL NUMBER RATIO                DB
C       AMAXA0             MAXIMUM 'A' CURVE EVALUATED AT A0      DB
C       AMAXA02            MAXIMUM 'A' CURVE EVALUATED AT A02     DB
C       AMAXB              MAXIMUM 'A' CURVE EVALUATED AT B       DB
C       AMINA              MINIMUM 'A' CURVE EVALUATED AT 
C                            STROUHAL NUMBER RATIO                DB
C       AMINA0             MINIMUM 'A' CURVE EVALUATED AT A0      DB
C       AMINA02            MINIMUM 'A' CURVE EVALUATED AT A02     DB
C       AMINB              MINIMUM 'A' CURVE EVALUATED AT B       DB
C       ARA0               INTERPOLATION FACTOR                  ---
C       ARA02              INTERPOLATION FACTOR                  ---
C       B                  STROUHAL NUMBER RATIO                 ---
C       B0                 FUNCTION USED IN 'B' CALCULATION      ---
C       BB                 'B' SPECTRUM SHAPE EVALUATED AT
C                            STROUHAL NUMBER RATIO                DB
C       BETA               USED IN 'B' COMPUTATION               ---
C       BETA0              USED IN 'B' COMPUTATION               ---
C       BMAXB              MAXIMUM 'B' EVALUATED AT B             DB
C       BMAXB0             MAXIMUM 'B' EVALUATED AT B0            DB
C       BMINB              MINIMUM 'B' EVALUATED AT B             DB
C       BMINB0             MINIMUM 'B' EVALUATED AT B0            DB
C       BRB0               INTERPOLATION FACTOR                   DB
C       C                  CHORD LENGTH                          METERS
C       C0                 SPEED OF SOUND                      METERS/SEC
C       DBARH              HIGH FREQUENCY DIRECTIVITY             ---
C       DBARL              LOW FREQUENCY DIRECTIVITY              ---
C       DELK1              CORRECTION TO AMPLITUDE FUNCTION       DB
C       DELTAP             PRESSURE SIDE BOUNDARY LAYER THICKNESS METERS
C       DSTRP              PRESSURE SIDE DISPLACEMENT THICKNESS  METERS
C       DSTRS              SUCTION SIDE DISPLACEMENT THICKNESS   METERS
C       FRCEN              ARRAY OF CENTERED FREQUENCIES         HERTZ
C       GAMMA              USED IN 'B' COMPUTATION                ---
C       GAMMA0             USED IN 'B' COMPUTATION                ---
C       ITRIP              TRIGGER TO TRIP BOUNDARY LAYER         ---
C       K1                 AMPLITUDE FUNCTION                     DB
C       K2                 AMPLITUDE FUNCTION                     DB
C       L                  SPAN                                  METERS
C       M                  MACH NUMBER                            ---
C       NFREQ              NUMBER OF CENTERED FREQUENCIES         ---
C       PHI                DIRECTIVITY ANGLE                    DEGREES
C       P1                 PRESSURE SIDE PRESSURE               NT/M2
C       P2                 SUCTION SIDE PRESSURE                NT/M2
C       P4                 PRESSURE FROM ANGLE OF ATTACK
C                            CONTRIBUTION                       NT/M2
C       R                  SOURCE TO OBSERVER DISTANCE           METERS
C       RC                 REYNOLDS NUMBER BASED ON  CHORD        ---
C       RDSTRP             REYNOLDS NUMBER BASED ON PRESSURE
C                            SIDE DISPLACEMENT THICKNESS          ---
C       RDSTRS             REYNOLDS NUMBER BASED ON SUCTION
C                            SIDE DISPLACEMENT THICKNESS          ---
C       SPLALPH            SOUND PRESSURE LEVEL DUE TO ANGLE OF 
C                            ATTACK CONTRIBUTION                  DB
C       SPLP               SOUND PRESSURE LEVEL DUE TO PRESSURE
C                            SIDE OF AIRFOIL                      DB
C       SPLS               SOUND PRESSURE LEVEL DUE TO SUCTION
C                            SIDE OF AIRFOIL                      DB
C       SPLTBL             TOTAL SOUND PRESSURE LEVEL DUE TO 
C                            TBLTE MECHANISM                      DB
C       STP                PRESSURE SIDE STROUHAL NUMBER          ---
C       STS                SUCTION SIDE STROUHAL NUMBER           ---
C       ST1                PEAK STROUHAL NUMBER                   ---
C       ST1PRIM            PEAK STROUHAL NUMBER                   ---
C       ST2                PEAK STROUHAL NUMBER                   ---
C       STPEAK             PEAK STROUHAL NUMBER                   ---
C       SWITCH             LOGICAL FOR COMPUTATION OF ANGLE 
C                            OF ATTACK CONTRIBUTION               ---
C       THETA              DIRECTIVITY ANGLE                     DEGREES
C       U                  VELOCITY                             METERS/SEC
C       VISC               KINEMATIC VISCOSITY                   M2/SEC
C       XCHECK             USED TO CHECK FOR ANGLE OF ATTACK
C                            CONTRIBUTION                         ---
C       

      USE Third_OCtave_Bands


      DIMENSION SPLTBL(NumBands)  ,SPLP(NumBands)    ,SPLS(NumBands)  ,
     1          SPLALPH(NumBands) ,STP(NumBands)     ,
     1          STS(NumBands)     ,FRCEN(NumBands)

      LOGICAL SWITCH
      INTEGER X_BLMethod
      REAL    L,M,K1,K2

      RC       = U  * C / VISC
      M        = U  / C0


C      COMPUTE BOUNDARY LAYER THICKNESSES
C      ----------------------------------

      SELECT CASE (X_BLMethod) 
      CASE (2)
	   CALL XTHICK_CALC(DELTAP,DSTRS,DSTRP)
	CASE DEFAULT 
         CALL THICK(C,U ,ALPSTAR,ITRIP,DELTAP,DSTRS,DSTRP,C0,VISC)
	END SELECT
c	write (5,*)"Delta*_Suction Delta*_Pressure"
c	write(5,*) dstrs, dstrp

C     COMPUTE DIRECTIVITY FUNCTION
C     ----------------------------

      CALL DIRECTL(M,THETA,PHI,DBARL)
      CALL DIRECTH(M,THETA,PHI,DBARH)


C     CALCULATE THE REYNOLDS NUMBERS BASED ON PRESSURE AND
C     SUCTION DISPLACEMENT THICKNESS
C     ---------------------------------------------------

      RDSTRS = DSTRS * U  / VISC
      RDSTRP = DSTRP * U  / VISC

C      DETERMINE PEAK STROUHAL NUMBERS TO BE USED FOR
C      'A' AND 'B' CURVE CALCULATIONS
C      ----------------------------------------------

      ST1    = .02 * M ** (-.6)

      IF (ALPSTAR .LE. 1.333) ST2 = ST1
      IF ((ALPSTAR .GT. 1.333).AND.(ALPSTAR .LE. 12.5))
     1   ST2 = ST1*10.**(.0054*(ALPSTAR-1.333)**2.)
      IF (ALPSTAR .GT. 12.5) ST2 = 4.72 * ST1


      ST1PRIM = (ST1+ST2)/2.


      CALL A0COMP(RC,A0)
      CALL A0COMP(3.*RC,A02)

C      EVALUATE MINIMUM AND MAXIMUM 'A' CURVES AT A0
C      ----------------------------------------------

      CALL AMIN(A0,AMINA0)
      CALL AMAX(A0,AMAXA0)

      CALL AMIN(A02,AMINA02)
      CALL AMAX(A02,AMAXA02)

C      COMPUTE 'A' MAX/MIN RATIO
C      -------------------------

      ARA0  = (20. + AMINA0) / (AMINA0 - AMAXA0)
      ARA02 = (20. + AMINA02)/ (AMINA02- AMAXA02)

C      COMPUTE B0 TO BE USED IN 'B' CURVE CALCULATIONS
C      -----------------------------------------------

      IF (RC .LT. 9.52E+04) B0 = .30
      IF ((RC .GE. 9.52E+04).AND.(RC .LT. 8.57E+05))
     1    B0 = (-4.48E-13)*(RC-8.57E+05)**2. + .56
      IF (RC .GE. 8.57E+05) B0 = .56

C      EVALUATE MINIMUM AND MAXIMUM 'B' CURVES AT B0
C      ----------------------------------------------

      CALL BMIN(B0,BMINB0)
      CALL BMAX(B0,BMAXB0)

C      COMPUTE 'B' MAX/MIN RATIO
C      -------------------------

      BRB0  = (20. + BMINB0) / (BMINB0 - BMAXB0)

C      FOR EACH CENTER FREQUENCY, COMPUTE AN
C      'A' PREDICTION FOR THE PRESSURE SIDE
C      -------------------------------------

      STPEAK = ST1

      DO 100 I=1,NFREQ
        STP(I) = FRCEN(I) * DSTRP / U 
        A      = ALOG10( STP(I) / STPEAK )
        CALL AMIN(A,AMINA)
        CALL AMAX(A,AMAXA)
        AA     = AMINA + ARA0 * (AMAXA - AMINA)

        IF (RC .LT. 2.47E+05) K1 = -4.31 * ALOG10(RC) + 156.3
        IF((RC .GE. 2.47E+05).AND.(RC .LT. 8.0E+05))
     1    K1 = -9.0 * ALOG10(RC) + 181.6
        IF (RC .GT. 8.0E+05) K1 = 128.5

        IF (RDSTRP .LE. 5000.) DELK1 = -ALPSTAR*(5.29-1.43*
     1    ALOG10(RDSTRP))
        IF (RDSTRP .GT. 5000.) DELK1 = 0.0

        SPLP(I)=AA+K1-3.+10.*ALOG10(DSTRP*M**5.*DBARH*L/R**2.)+DELK1




      GAMMA   = 27.094 * M +  3.31
      BETA    = 72.650 * M + 10.74
      GAMMA0  = 23.430 * M +  4.651
      BETA0   =-34.190 * M - 13.820

      IF (ALPSTAR .LE. (GAMMA0-GAMMA)) K2 = -1000.0
      IF ((ALPSTAR.GT.(GAMMA0-GAMMA)).AND.(ALPSTAR.LE.(GAMMA0+GAMMA)))
     1 K2=SQRT(BETA**2.-(BETA/GAMMA)**2.*(ALPSTAR-GAMMA0)**2.)+BETA0
      IF (ALPSTAR .GT. (GAMMA0+GAMMA)) K2 = -12.0

      K2 = K2 + K1



      STS(I) = FRCEN(I) * DSTRS / U 

C      CHECK FOR 'A' COMPUTATION FOR SUCTION SIDE
C      ------------------------------------------

      XCHECK = GAMMA0
      SWITCH = .FALSE.
      IF ((ALPSTAR .GE. XCHECK).OR.(ALPSTAR .GT. 12.5))SWITCH=.TRUE.
      IF (.NOT. SWITCH) THEN
        A      = ALOG10( STS(I) / ST1PRIM )
        CALL AMIN(A,AMINA)
        CALL AMAX(A,AMAXA)
        AA = AMINA + ARA0 * (AMAXA - AMINA)

        SPLS(I) = AA+K1-3.+10.*ALOG10(DSTRS*M**5.*DBARH*
     1            L/R**2.) 

C      'B' CURVE COMPUTATION
C       --------------------

        B = ABS(ALOG10(STS(I) / ST2))
        CALL BMIN(B,BMINB)
        CALL BMAX(B,BMAXB)
        BB = BMINB + BRB0 * (BMAXB-BMINB)
        SPLALPH(I)=BB+K2+10.*ALOG10(DSTRS*M**5.*DBARH*L/R**2.) 

      ELSE

C       THE 'A' COMPUTATION IS DROPPED IF 'SWITCH' IS TRUE
C       --------------------------------------------------


        SPLS(I) = 0.0 + 10.*ALOG10(DSTRS*M**5.*DBARL*
     1              L/R**2.) 
        SPLP(I) = 0.0 + 10.*ALOG10(DSTRS*M**5.*DBARL*
     1              L/R**2.) 
        B = ABS(ALOG10(STS(I) / ST2))
        CALL AMIN(B,AMINB)
        CALL AMAX(B,AMAXB)
        BB = AMINB + ARA02 * (AMAXB-AMINB)
        SPLALPH(I)=BB+K2+10.*ALOG10(DSTRS*M**5.*DBARL*
     1           L/R**2.)  
      ENDIF


C      SUM ALL CONTRIBUTIONS FROM 'A' AND 'B' ON BOTH 
C      PRESSURE AND SUCTION SIDE ON A MEAN-SQUARE PRESSURE
C      BASIS
C      ---------------------------------------------------

      IF (SPLP(I)    .LT. -100.) SPLP(I)    = -100.
      IF (SPLS(I)    .LT. -100.) SPLS(I)    = -100.
      IF (SPLALPH(I) .LT. -100.) SPLALPH(I) = -100.

      P1  = 10.**(SPLP(I) / 10.)
      P2  = 10.**(SPLS(I) / 10.)
      P4  = 10.**(SPLALPH(I) / 10.)

      SPLTBL(I) = 10. * ALOG10(P1 + P2 + P4)

  100 CONTINUE

      RETURN
      END
      
      SUBROUTINE AMIN(A,AMINA)

C     THIS SUBROUTINE DEFINES THE CURVE FIT CORRESPONDING
C     TO THE A-CURVE FOR THE MINIMUM ALLOWED REYNOLDS NUMBER.
C     

      X1 = ABS(A)
    
      IF (X1 .LE. .204) AMINA=SQRT(67.552-886.788*X1**2.)-8.219
      IF((X1 .GT. .204).AND.(X1 .LE. .244))AMINA=-32.665*X1+3.981
      IF (X1 .GT. .244)AMINA=-142.795*X1**3.+103.656*X1**2.-57.757*X1+6.006

      RETURN
      END
      SUBROUTINE AMAX(A,AMAXA)

C     THIS SUBROUTINE DEFINES THE CURVE FIT CORRESPONDING
C     TO THE A-CURVE FOR THE MAXIMUM ALLOWED REYNOLDS NUMBER.

      X1 = ABS(A)

      IF (X1 .LE. .13)AMAXA=SQRT(67.552-886.788*X1**2.)-8.219
      IF((X1 .GT. .13).AND.(X1 .LE. .321))AMAXA=-15.901*X1+1.098
      IF (X1 .GT. .321)AMAXA=-4.669*X1**3.+3.491*X1**2.-16.699*X1+1.149

      RETURN
      END
      SUBROUTINE BMIN(B,BMINB)

C     THIS SUBROUTINE DEFINES THE CURVE FIT CORRESPONDING 
C     TO THE B-CURVE FOR THE MINIMUM ALLOWED REYNOLDS NUMBER.

      X1 = ABS(B)
   
      IF (X1 .LE. .13)BMINB=SQRT(16.888-886.788*X1**2.)-4.109
      IF((X1 .GT. .13).AND.(X1 .LE. .145))BMINB=-83.607*X1+8.138
      IF (X1.GT..145)BMINB=-817.81*X1**3.+355.21*X1**2.-135.024*X1+10.619

      RETURN
      END
      SUBROUTINE BMAX(B,BMAXB)

C     THIS SUBROUTINE DEFINES THE CURVE FIT CORRESPONDING
C     TO THE B-CURVE FOR THE MAXIMUM ALLOWED REYNOLDS NUMBER.

      X1 = ABS(B)

      IF (X1 .LE. .1) BMAXB=SQRT(16.888-886.788*X1**2.)-4.109
      IF((X1 .GT. .1).AND.(X1 .LE. .187))BMAXB=-31.313*X1+1.854
      IF (X1.GT..187)BMAXB=-80.541*X1**3.+44.174*X1**2.-39.381*X1+2.344

      RETURN
      END
      SUBROUTINE A0COMP(RC,A0) 

C     THIS SUBROUTINE DETERMINES WHERE THE A-CURVE 
C     TAKES ON A VALUE OF -20 dB.

      IF (RC .LT. 9.52E+04) A0 = .57
      IF ((RC .GE. 9.52E+04).AND.(RC .LT. 8.57E+05))
     1   A0 = (-9.57E-13)*(RC-8.57E+05)**2. + 1.13
      IF (RC .GE. 8.57E+05) A0 = 1.13
      RETURN
      END
      SUBROUTINE DIRECTH(M,THETA,PHI,DBAR)

C     THIS SUBROUTINE COMPUTES THE HIGH FREQUENCY
C     DIRECTIVITY FUNCTION FOR THE INPUT OBSERVER LOCATION

      REAL M,MC

      DEGRAD  = .017453

      MC     = .8 * M
      THETAR = THETA * DEGRAD
      PHIR   = PHI * DEGRAD

      DBAR=2.*SIN(THETAR/2.)**2.*SIN(PHIR)**2./((1.+M*COS(THETAR))*
     1      (1.+(M-MC)*COS(THETAR))**2.)
      RETURN
      END
      SUBROUTINE DIRECTL(M,THETA,PHI,DBAR)

C     THIS SUBROUTINE COMPUTES THE LOW FREQUENCY
C     DIRECTIVITY FUNCTION FOR THE INPUT OBSERVER LOCATION

      REAL M,MC

      DEGRAD  = .017453

      MC     = .8 * M
      THETAR = THETA * DEGRAD
      PHIR   = PHI * DEGRAD

      DBAR = (SIN(THETAR)*SIN(PHIR))**2/(1.+M*COS(THETAR))**4

      RETURN
      END
      SUBROUTINE BLUNT(ALPSTAR,C,U ,FRCEN,ITRIP,SPLBLNT,THETA,PHI,
     1                 L,R,H,PSI,NFREQ,VISC,C0,X_BLMethod)


C                  --------------------------------
C                  ***** VARIABLE DEFINITIONS *****
C                  --------------------------------

C       VARIABLE NAME               DEFINITION                  UNITS
C       -------------               ----------                  -----

C       ALPSTAR            ANGLE OF ATTACK                     DEGREES
C       ATERM              USED TO COMPUTE PEAK STROUHAL NO.    ---
C       C                  CHORD LENGTH                        METERS
C       C0                 SPEED OF SOUND                      METERS/SEC
C       DBARH              HIGH FREQUENCY DIRECTIVITY           ---
C       DELTAP             PRESSURE SIDE BOUNDARY LAYER 
C                            THICKNESS                          METERS
C       DSTARH             AVERAGE DISPLACEMENT THICKNESS
C                            OVER TRAILING EDGE BLUNTNESS       ---
C       DSTRAVG            AVERAGE DISPLACEMENT THICKNESS       METERS
C       DSTRP              PRESSURE SIDE DISPLACEMENT THICKNESS METERS
C       DSTRS              SUCTION SIDE DISPLACEMENT THICKNESS  METERS
C       ETA                RATIO OF STROUHAL NUMBERS             ---
C       FRCEN              ARRAY OF 1/3 OCTAVE CENTERED FREQ.   HERTZ
C       F4TEMP             G5 EVALUATED AT MINIMUM HDSTARP       DB
C       G4                 SCALED SPECTRUM LEVEL                 DB
C       G5                 SPECTRUM SHAPE FUNCTION               DB
C       G50                G5 EVALUATED AT PSI=0.0               DB
C       G514               G5 EVALUATED AT PSI=14.0              DB
C       H                  TRAILING EDGE BLUNTNESS              METERS
C       HDSTAR             BLUNTNESS OVER AVERAGE DISPLACEMENT 
C                            THICKNESS                           ---
C       HDSTARL            MINIMUM ALLOWED VALUE OF HDSTAR       ---
C       HDSTARP            MODIFIED VALUE OF HDSTAR              ---
C       ITRIP              TRIGGER FOR BOUNDARY LAYER TRIPPING    ---
C       L                  SPAN                                  METERS
C       M                  MACH NUMBER                           ---
C       NFREQ              NUMBER OF CENTERED FREQUENCIES        ---
C       PHI                DIRECTIVITY ANGLE                    DEGREES
C       PSI                TRAILING EDGE ANGLE                  DEGREES
C       R                  SOURCE TO OBSERVER DISTANCE           METERS
C       RC                 REYNOLDS NUMBER BASED ON CHORD        ---
C       SCALE              SCALING FACTOR                        ---
C       SPLBLNT            SOUND PRESSURE LEVELS DUE TO 
C                            BLUNTNESS                            DB
C       STPEAK             PEAK STROUHAL NUMBER                  ---
C       STPPP              STROUHAL NUMBER                       ---
C       THETA              DIRECTIVITY ANGLE                     ---
C       U                  FREESTREAM VELOCITY                 METERS/SEC
C       VISC               KINEMATIC VISCOSITY                 M2/SEC

      
      USE Third_OCtave_Bands
       
      DIMENSION SPLBLNT(NumBands)  ,FRCEN(NumBands)   ,STPPP(NumBands)

      REAL M,L
	INTEGER X_BLMethod

C      COMPUTE NECESSARY QUANTITIES
C      ----------------------------

      M  = U /C0
      RC = U  * C / VISC


C      COMPUTE BOUNDARY LAYER THICKNESSES
C      ----------------------------------

      SELECT CASE (X_BLMethod) 
      CASE (2)
	   CALL XTHICK_CALC(DELTAP,DSTRS,DSTRP)
	CASE DEFAULT 
         CALL THICK(C,U ,ALPSTAR,ITRIP,DELTAP,DSTRS,DSTRP,C0,VISC)
	END SELECT

C      COMPUTE AVERAGE DISPLACEMENT THICKNESS
C      --------------------------------------

      DSTRAVG = (DSTRS + DSTRP) / 2.
      HDSTAR  = H / DSTRAVG
 
      DSTARH = 1. /HDSTAR

C      COMPUTE DIRECTIVITY FUNCTION
C      ----------------------------

      CALL DIRECTH(M,THETA,PHI,DBARH)


C      COMPUTE PEAK STROUHAL NUMBER
C      ----------------------------

      ATERM  = .212 - .0045 * PSI

      IF (HDSTAR .GE. .2)
     1   STPEAK    = ATERM / (1.+.235*DSTARH-.0132*DSTARH**2.)
      IF (HDSTAR .LT. .2) 
     1   STPEAK    = .1 * HDSTAR + .095 - .00243 * PSI

C      COMPUTE SCALED SPECTRUM LEVEL
C      -----------------------------

      IF (HDSTAR .LE. 5.) G4=17.5*ALOG10(HDSTAR)+157.5-1.114*PSI
      IF (HDSTAR .GT. 5.) G4=169.7 - 1.114 * PSI


C      FOR EACH FREQUENCY, COMPUTE SPECTRUM SHAPE REFERENCED TO 0 DB
C      -------------------------------------------------------------

      DO 1000 I=1,NFREQ

        STPPP(I) = FRCEN(I) * H / U 
        ETA      = ALOG10(STPPP(I)/STPEAK)

        HDSTARL = HDSTAR

        CALL G5COMP(HDSTARL,ETA,G514)

        HDSTARP = 6.724 * HDSTAR **2.-4.019*HDSTAR+1.107

        CALL G5COMP(HDSTARP,ETA,G50)


        G5 = G50 + .0714 * PSI * (G514-G50)
        IF (G5 .GT. 0.) G5 = 0.
        CALL G5COMP(.25,ETA,F4TEMP)
        IF (G5 .GT. F4TEMP) G5 = F4TEMP


        SCALE = 10. * ALOG10(M**5.5*H*DBARH*L/R**2.)

        SPLBLNT(I) = G4 + G5 + SCALE


 1000 CONTINUE

      RETURN  
      END
      SUBROUTINE G5COMP(HDSTAR,ETA,G5)


      REAL M,K,MU

      
      IF (HDSTAR .LE. .25) MU = .1211
      IF ((HDSTAR .GT. .25).AND.(HDSTAR .LE. .62))
     1     MU=-.2175*HDSTAR + .1755
      IF ((HDSTAR .GT. .62).AND.(HDSTAR .LT. 1.15))
     1 MU = -.0308 * HDSTAR + .0596
      IF (HDSTAR .GE. 1.15)MU = .0242

      IF (HDSTAR .LE. .02) M = 0.0
      IF ((HDSTAR .GE. .02).AND.(HDSTAR .LT. .5))
     1    M=68.724*HDSTAR - 1.35     
      IF ((HDSTAR .GE. .5).AND.(HDSTAR .LE. .62))
     1  M = 308.475 * HDSTAR - 121.23
      IF ((HDSTAR .GT. .62).AND.(HDSTAR .LE. 1.15))
     1  M = 224.811 * HDSTAR - 69.354
      IF ((HDSTAR .GT. 1.15) .AND. (HDSTAR .LT. 1.2))
     1  M = 1583.28 * HDSTAR - 1631.592
      IF (HDSTAR .GE. 1.2) M = 268.344
      IF (M .LT. 0.0) M = 0.0

      ETA0 = -SQRT((M*M*MU**4)/(6.25+M*M*MU*MU))

      K    = 2.5*SQRT(1.-(ETA0/MU)**2.)-2.5-M*ETA0

      IF (ETA .LE. ETA0) G5 = M * ETA + K
      IF ((ETA .GT. ETA0).AND.(ETA .LE. 0.))G5=2.5*SQRT(1.-(ETA/MU)**2.)-2.5
      IF((ETA.GT.0.).AND.(ETA.LE..03616))G5=SQRT(1.5625-1194.99*ETA**2.)-1.25
      IF (ETA .GT. .03616) G5=-155.543 * ETA + 4.375

      RETURN
      END
      SUBROUTINE TIPNOIS(ALPHTIP,ALPRAT,C,U ,FRCEN,SPLTIP,THETA,PHI,
     1                   R,NFREQ,VISC,C0,ROUND)

C                  --------------------------------
C                  ***** VARIABLE DEFINITIONS *****
C                  --------------------------------

C       VARIABLE NAME               DEFINITION                  UNITS
C       -------------               ----------                  -----

C       ALPHTIP            TIP ANGLE OF ATTACK                DEGREES
C       ALPRAT             TIP LIFT CURVE SLOPE                 ---
C       ALPTIPP            CORRECTED TIP ANGLE OF ATTACK      DEGREES
C       C                  CHORD LENGTH                         METERS
C       C0                 SPEED OF SOUND                    METERS/SEC
C       DBARH              DIRECTIVITY                         ---
C       FRCEN              CENTERED FREQUENCIES              HERTZ
C       L                  CHARACTERISTIC LENGTH FOR TIP      METERS
C       M                  MACH NUMBER                         ---
C       MM                 MAXIMUM MACH NUMBER                 ---
C       NFREQ              NUMBER OF CENTERED FREQUENCIES      ---
C       PHI                DIRECTIVITY ANGLE                  DEGREES
C       R                  SOURCE TO OBSERVER DISTANCE        METERS
C       ROUND              LOGICAL SET TRUE IF TIP IS ROUNDED  ---
C       SCALE              SCALING TERM                        ---
C       SPLTIP             SOUND PRESSURE LEVEL DUE TO TIP
C                            MECHANISM                         DB
C       STPP               STROUHAL NUMBER                     ---
C       TERM               SCALING TERM                        ---
C       THETA              DIRECTIVITY ANGLE                  DEGREES
C       U                  FREESTREAM VELOCITY               METERS/SEC
C       UM                 MAXIMUM VELOCITY                  METERS/SEC
C       VISC               KINEMATIC VISCOSITY               M2/SEC

      USE Third_Octave_Bands

      DIMENSION SPLTIP(NumBands),FRCEN(NumBands)
      REAL L,M,MM
      LOGICAL ROUND


      ALPTIPP = ALPHTIP * ALPRAT
      M       = U  / C0

      CALL DIRECTH(M,THETA,PHI,DBARH)

      IF (ROUND) THEN
        L = .008 * ALPTIPP * C
      ELSE
        IF (ABS(ALPTIPP) .LE. 2.) THEN
          L = (.023 + .0169*ALPTIPP) * C
        ELSE
          L = (.0378 + .0095*ALPTIPP) * C
        ENDIF
      ENDIF
        

      MM     = (1. + .036*ALPTIPP) * M

      UM     = MM * C0

      TERM  = M*M*MM**3.*L**2.*DBARH/R**2.
      IF (TERM .NE. 0.0) THEN
        SCALE = 10.*ALOG10(TERM)
      ELSE
        SCALE = 0.0
      ENDIF

      DO 100 I=1,NFREQ
        STPP      = FRCEN(I) * L / UM
        SPLTIP(I) = 126.-30.5*(ALOG10(STPP)+.3)**2. + SCALE
  100 CONTINUE
      RETURN
      END
      SUBROUTINE THICK(C,U ,ALPSTAR,ITRIP,DELTAP,DSTRS,DSTRP,C0,VISC)

C                  --------------------------------
C                  ***** VARIABLE DEFINITIONS *****
C                  --------------------------------

C       VARIABLE NAME               DEFINITION                  UNITS
C       -------------               ----------                  -----

C       ALPSTAR            ANGLE OF ATTACK                    DEGREES
C       C                  CHORD LENGTH                        METERS
C       C0                 SPEED OF SOUND                    METERS/SEC
C       DELTA0             BOUNDARY LAYER THICKNESS AT
C                            ZERO ANGLE OF ATTACK              METERS
C       DELTAP             PRESSURE SIDE BOUNDARY LAYER
C                            THICKNESS                         METERS
C       DSTR0              DISPLACEMENT THICKNESS AT ZERO
C                            ANGLE OF ATTACK                   METERS
C       DSTRP              PRESSURE SIDE DISPLACEMENT 
C                            THICKNESS                         METERS
C       DSTRS              SUCTION SIDE DISPLACEMENT 
C                            THICKNESS                         METERS
C       ITRIP              TRIGGER FOR BOUNDARY LAYER TRIPPING  ---
C       M                  MACH NUMBER                          ---
C       RC                 REYNOLDS NUMBER BASED ON CHORD       ---
C       U                  FREESTREAM VELOCITY                METERS/SEC
C       VISC               KINEMATIC VISCOSITY                M2/SEC


C      COMPUTE ZERO ANGLE OF ATTACK BOUNDARY LAYER
C      THICKNESS (METERS) AND REYNOLDS NUMBER
C      -------------------------------------------

      M        = U  / C0

      RC       = U  * C/VISC

      DELTA0   = 10.**(1.6569-.9045*ALOG10(RC)+
     1           .0596*ALOG10(RC)**2.)*C
      IF (ITRIP .EQ. 2) DELTA0 = .6 * DELTA0


C      COMPUTE PRESSURE SIDE BOUNDARY LAYER THICKNESS
C      ----------------------------------------------

      DELTAP   = 10.**(-.04175*ALPSTAR+.00106*ALPSTAR**2.)*DELTA0


C      COMPUTE ZERO ANGLE OF ATTACK DISPLACEMENT THICKNESS
C      ---------------------------------------------------

      IF ((ITRIP .EQ. 1) .OR. (ITRIP .EQ. 2)) THEN
        IF (RC .LE. .3E+06) DSTR0 = .0601 * RC **(-.114)*C
        IF (RC .GT. .3E+06) 
     1    DSTR0=10.**(3.411-1.5397*ALOG10(RC)+.1059*ALOG10(RC)**2.)*C
        IF (ITRIP .EQ. 2) DSTR0 = DSTR0 * .6
      ELSE
        DSTR0=10.**(3.0187-1.5397*ALOG10(RC)+.1059*ALOG10(RC)**2.)*C
      ENDIF

C      PRESSURE SIDE DISPLACEMENT THICKNESS
C      ------------------------------------

      DSTRP   = 10.**(-.0432*ALPSTAR+.00113*ALPSTAR**2.)*DSTR0
      IF (ITRIP .EQ. 3) DSTRP = DSTRP * 1.48

C      SUCTION SIDE DISPLACEMENT THICKNESS
C      -----------------------------------

      IF (ITRIP .EQ. 1) THEN
        IF (ALPSTAR .LE. 5.) DSTRS=10.**(.0679*ALPSTAR)*DSTR0
        IF((ALPSTAR .GT. 5.).AND.(ALPSTAR .LE. 12.5))
     1   DSTRS = .381*10.**(.1516*ALPSTAR)*DSTR0
        IF (ALPSTAR .GT. 12.5)DSTRS=14.296*10.**(.0258*ALPSTAR)*DSTR0
      ELSE
        IF (ALPSTAR .LE. 7.5)DSTRS =10.**(.0679*ALPSTAR)*DSTR0
        IF((ALPSTAR .GT. 7.5).AND.(ALPSTAR .LE. 12.5))
     1   DSTRS = .0162*10.**(.3066*ALPSTAR)*DSTR0
        IF (ALPSTAR .GT. 12.5) DSTRS = 52.42*10.**(.0258*ALPSTAR)*DSTR0
      ENDIF
      
      RETURN
      END
