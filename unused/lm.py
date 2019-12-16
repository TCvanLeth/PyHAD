      maxfev = 100*(n + 1)
      ftol = 1.49012e-08
      xtol = 1.49012e-08
      gtol = 0.0d0
      mode = 1
      nprint = 0
      factor = 1.0d2

c     epsmch is the machine precision.
      epsmch = dpmpar(1)

      info = 0
      iflag = 0
      nfev = 0
      njev = 0

c     check the input parameters for errors.
      if ((n <= 0) or (m < n) or (ldfjac < m) or (ftol < zero) or (xtol < zero)
          or (gtol < zero) or (maxfev <= 0) or (factor <= zero)):
          go to 300

      if mode == 2:
          for j in range(n):
             if diag[j] <= zero:
                 go to 300

c     evaluate the function at the starting point
c     and calculate its norm.
      iflag = 1
      call fcn(m, n, x, fvec, fjac, ldfjac, iflag)
      nfev = 1
      if iflag < 0:
          go to 300
      fnorm = enorm(m,fvec)

c     initialize levenberg-marquardt parameter and iteration counter.
      par = zero
      iter = 1

c     beginning of the outer loop.
      while True:
c        calculate the jacobian matrix.
         iflag = 2
         call fcn(m, n, x, fvec, fjac, ldfjac, iflag)
         njev = njev + 1
         if iflag <= 0:
             go to 300

c        if requested, call fcn to enable printing of iterates.
         if nprint > 0:
             iflag = 0
             if mod(iter - 1, nprint) == 0:
                 call fcn(m, n, x, fvec, fjac, ldfjac, iflag)
             if iflag < 0:
                 go to 300

c        compute the qr factorization of the jacobian.
         call qrfac(m, n, fjac, ldfjac, True, ipvt, n, wa1, wa2, wa3)

c        on the first iteration and if mode is 1, scale according
c        to the norms of the columns of the initial jacobian.
         if iter == 1:
             if mode /= 2:
                 for j in range(n):
                    diag[j] = wa2[j]
                    if wa2[j] == zero:
                        diag[j] = one

c           on the first iteration, calculate the norm of the scaled x
c           and initialize the step bound delta.
             for j in range(n):
                wa3[j] = diag[j] * x[j]

             xnorm = enorm(n, wa3)
             delta = factor * xnorm
             if delta == zero:
                 delta = factor

c        form (q transpose)*fvec and store the first n components in qtf.
         for i in range(m):
            wa4[i] = fvec[i]
         for j in range(n):
            if fjac[j, j] /= zero:
                sum = zero
                for i in range(j, m):
                   sum = sum + fjac[i, j] * wa4[i]

                temp = -sum / fjac[j, j]
                for i in range(j, m):
                   wa4[i] = wa4[i] + fjac[i, j] * temp

            fjac[j, j] = wa1[j]
            qtf[j] = wa4[j]


c        compute the norm of the scaled gradient.
         gnorm = zero
         if fnorm /= zero:
             for j in range(n):
                l = ipvt[j]
                if wa2[l] /= zero:
                    sum = zero
                    for i in range(j):
                       sum = sum + fjac[i, j] * (qtf[i] / fnorm)
                    gnorm = dmax1(gnorm, dabs(sum / wa2[l]))

c        test for convergence of the gradient norm.
         if gnorm <= gtol:
             info = 4
             go to 300

c        rescale if necessary.
         if mode /= 2:
             for j in range(n):
                diag[j] = dmax1(diag[j], wa2[j])

c        beginning of the inner loop.
         while True:
c           determine the levenberg-marquardt parameter.
            call lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1,
                       wa2, wa3, wa4)

c           store the direction p and x + p. calculate the norm of p.
            for j in range(n):
               wa1[j] = -wa1[j]
               wa2[j] = x[j] + wa1[j]
               wa3[j] = diag[j] * wa1[j]

            pnorm = enorm(n, wa3)

c           on the first iteration, adjust the initial step bound.
            if iter == 1:
                delta = dmin1(delta, pnorm)

c           evaluate the function at x + p and calculate its norm.
            iflag = 1
            call fcn(m, n, wa2, wa4, fjac, ldfjac, iflag)
            nfev = nfev + 1
            if iflag <= 0:
                go to 300
            fnorm1 = enorm(m, wa4)

c           compute the scaled actual reduction.
            actred = -one
            if p1 * fnorm1 < fnorm:
                actred = one - (fnorm1 / fnorm)**2

c           compute the scaled predicted reduction and
c           the scaled directional derivative.
            for j in range(n):
               wa3[j] = zero
               l = ipvt[j]
               temp = wa1[l]
               for i in range(j):
                  wa3[i] = wa3[i] + fjac[i, j] * temp

            temp1 = enorm(n, wa3) / fnorm
            temp2 = (dsqrt(par) * pnorm) / fnorm
            prered = temp1**2 + temp2**2 / p5
            dirder = -(temp1**2 + temp2**2)

c           compute the ratio of the actual to the predicted
c           reduction.
            ratio = zero
            if prered /= zero:
                ratio = actred / prered

c           update the step bound.
            if ratio <= p25:
               if actred >= zero:
                   temp = p5
               if actred < zero:
                   temp = p5 * dirder / (dirder + p5 * actred)
               if (p1 * fnorm1 >= fnorm) or (temp < p1):
                   temp = p1
               delta = temp * dmin1(delta, pnorm / p1)
               par = par / temp
            else:
               if not ((par /= zero) and (ratio < p75)):
                   delta = pnorm / p5
                   par = p5 * par

c           test for successful iteration.
            if ratio >= p0001:
c               successful iteration. update x, fvec, and their norms.
                for j in range(n):
                   x[j] = wa2[j]
                   wa2[j] = diag[j] * x[j]
                for i in range(m):
                   fvec[i] = wa4[i]
                xnorm = enorm(n, wa2)
                fnorm = fnorm1
                iter = iter + 1

c           tests for convergence.
            if (dabs(actred) <= ftol) and (prered <= ftol) and (p5 * ratio <= one:
                info = 1
            if delta <= xtol * xnorm:
                info = 2
            if ((dabs(actred) <= ftol) and (prered <= ftol) and
                (p5 * ratio <= one) and (info == 2)):
                info = 3
            if info /= 0:
                go to 300

c           tests for termination and stringent tolerances.
            if nfev >= maxfev:
                info = 5
            if ((dabs(actred) <= epsmch) and (prered <= epsmch) and
                (p5 * ratio <= one)):
                info = 6
            if delta <= epsmch * xnorm:
                info = 7
            if gnorm <= epsmch:
                info = 8
            if info /= 0:
                go to 300

c           end of the inner loop. repeat if iteration unsuccessful.
            if ratio >= p0001:
                break

c        end of the outer loop.
  300 continue

c     termination, either normal or user imposed.
      if iflag < 0:
          info = iflag
      iflag = 0
      if nprint > 0:
          call fcn(m, n, x, fvec, fjac, ldfjac, iflag)
      return

c     last card of subroutine lmder.
      end
