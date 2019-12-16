import numpy as np


def outer(func, x, m, n, diag=None, ldfjac):
    maxfev = 100*(n + 1)
    ftol = 1.49012e-08
    xtol = 1.49012e-08
    gtol = 0.0
    nprint = 0
    factor = 100.

    iflag = 0
    x, fvec, nfev, fjac, ipvt, qtf, info, iflag = inner(func, x, m, n, diag,
                                                        factor, maxfev, ldfjac,
                                                        ftol, xtol, gtol, nprint)

    # termination, either normal or user imposed.
    if iflag < 0:
        info = iflag
    iflag = 0
    if nprint > 0:
        fvec, fjac = func(m, n, x, ldfjac, iflag)
    return x, fvec, nfev, fjac, ipvt, qtf, info


def inner(func, x, m, n, diag, factor, maxfev, ldfjac, ftol, xtol, gtol, nprint):
    wa3 = np.empty(n)
    wa4 = np.empty(m)
    qtf = np.empty(n)
    ipvt = np.empty(n, dtype=int)

    info = 0
    nfev = 0
    njev = 0

    # epsmch is the machine precision.
    epsmch = dpmpar(1)

    # check the input parameters for errors.
    if ((n <= 0) or (m < n) or (ldfjac < m) or (ftol < 0.) or (xtol < 0.)
        or (gtol < 0.) or (maxfev <= 0) or (factor <= 0.)):
          return x, np.zeros(m), nfev, np.zeros((ldfjac, n)), ipvt, qtf, info

    if diag is not None and np.any(diag <= 0):
        return x, np.zeros(m), nfev, np.zeros((ldfjac, n)), ipvt, qtf, info

    # evaluate the function at the starting point
    # and calculate its norm.
    iflag = 1
    fvec, fjac = func(m, n, x, ldfjac, iflag) # user function
    nfev = 1
    if iflag < 0:
        return x, fvec, nfev, fjac, ipvt, qtf, info
    fnorm = np.linalg.norm(fvec)

    # initialize levenberg-marquardt parameter and iteration counter.
    par = 0.
    iter = 0

    # beginning of the outer loop.
    while True:
        # calculate the jacobian matrix.
        iflag = 2
        fvec, fjac = func(m, n, x, fvec, fjac, ldfjac, iflag) # user function
        njev = njev + 1
        if iflag <= 0:
            break

        # if requested, call fcn to enable printing of iterates.
        if nprint > 0:
            iflag = 0
            if iter % nprint == 0:
                fvec, fjac = func(m, n, x, ldfjac, iflag) # user function
            if iflag < 0:
                break

        # compute the qr factorization of the jacobian.
        fjac, ipvt, rdiag, acnorm = qrfac(fjac, True)

        # on the first iteration and if diag is not given, scale according
        # to the norms of the columns of the initial jacobian.
        if iter == 0 and diag is None:
            diag = np.where(acnorm == 0., 1., acnorm)

        # on the first iteration, calculate the norm of the scaled x
        # and initialize the step bound delta.
        wa3 = diag * x
        xnorm = np.linalg.norm(wa3)
        delta = factor * xnorm
        if delta == 0.:
            delta = factor

        # form (q transpose)*fvec and store the first n components in qtf.
        ind = np.diag_indices_from(fjac)
        wa4 = fvec
        for j in range(n):
            if fjac[j, j] != 0.:
                temp[j] = -(fjac[j:, j] * wa4[j:]).sum() / fjac[j, j]
                wa4[j:] += fjac[j:, j] * temp[j]

        fjac[ind] = rdiag
        qtf = wa4

        # compute the norm of the scaled gradient.
        gnorm = 0.
        if fnorm != 0.:
            for j in range(n):
                l = ipvt[j]
                if acnorm[l] != 0.:
                    sum = (fjac[:, j] * (qtf / fnorm)).sum()
                    gnorm = max(gnorm, abs(sum / acnorm[l]))

        # test for convergence of the gradient norm.
        if gnorm <= gtol:
            info = 4
            break

        # rescale if necessary.
        if diag is None:
            diag= np.maximum(diag, acnorm)

        # beginning of the inner loop.
        while True:
            # determine the levenberg-marquardt parameter.
            fjac, par, wa1, wa2 = lmpar(n, fjac, ldfjac, ipvt, diag, qtf,
                                        delta, par)

            # store the direction p and x + p. calculate the norm of p.
            wa1 = -wa1
            wa2 = x + wa1
            wa3 = diag * wa1
            pnorm = np.linalg.norm(wa3)

            # on the first iteration, adjust the initial step bound.
            if iter == 0:
                delta = min(delta, pnorm)

            # evaluate the function at x + p and calculate its norm.
            iflag = 1
            wa4, fjac = func(m, n, wa2, ldfjac, iflag) # user function
            nfev = nfev + 1
            if iflag <= 0:
                return x, fvec, nfev, fjac, ipvt, qtf, info
            fnorm1 = np.linalg.norm(wa4)

            # compute the scaled actual reduction.
            actred = -1.
            if .1 * fnorm1 < fnorm:
                actred = 1. - (fnorm1 / fnorm)**2

            # compute the scaled predicted reduction and
            # the scaled directional derivative.
            for j in range(n):
                wa3[j] = 0.
                l = ipvt[j]
                temp = wa1[l]
                wa3[:j] += fjac[:j, j] * temp

            temp1 = np.linalg.norm(wa3) / fnorm
            temp2 = (np.sqrt(par) * pnorm) / fnorm
            prered = temp1**2 + temp2**2 / .5
            dirder = -(temp1**2 + temp2**2)

            # compute the ratio of the actual to the predicted reduction.
            ratio = 0.
            if prered != 0.:
                ratio = actred / prered

            # update the step bound.
            if ratio <= 0.25:
               if actred >= 0.0:
                   temp = 0.5
               if actred < 0.0:
                   temp = 0.5 * dirder / (dirder + 0.5 * actred)
               if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                   temp = 0.1
               delta = temp * min(delta, pnorm / 0.1)
               par = par / temp
            elif par == 0.0 or ratio >= 0.75:
                delta = pnorm / 0.5
                par = 0.5 * par

            # test for successful iteration.
            if ratio >= 1e-4:
                # successful iteration. update x, fvec, and their norms.
                x = wa2
                wa2 = diag * x
                fvec = wa4
                xnorm = np.linalg.norm(wa2)
                fnorm = fnorm1
                iter = iter + 1

            # tests for convergence.
            if (abs(actred) <= ftol) and (prered <= ftol) and (.5 * ratio <= 1.):
                info = 1
            if delta <= xtol * xnorm:
                info = 2
            if ((abs(actred) <= ftol) and (prered <= ftol) and
                (0.5 * ratio <= 1.) and (info == 2)):
                info = 3
            if info != 0:
                return x, fvec, nfev, fjac, ipvt, qtf, info

            # tests for termination and stringent tolerances.
            if nfev >= maxfev:
                info = 5
            if ((abs(actred) <= epsmch) and (prered <= epsmch) and
                (.5 * ratio <= 1.)):
                info = 6
            if delta <= epsmch * xnorm:
                info = 7
            if gnorm <= epsmch:
                info = 8
            if info != 0:
                return x, fvec, nfev, fjac, ipvt, qtf, info

            # end of the inner loop. repeat if iteration unsuccessful.
            if ratio >= 1e-4:
                break
    return x, fvec, nfev, fjac, ipvt, qtf, info


###############################################################################
def qrfac(a,  pivot):
    m, n = a.shape

    # epsmch is the machine precision.
    epsmch = dpmpar(1)

    # compute the initial column norms and initialize several arrays.
    acnorm = np.linalg.norm(a[0,:])
    rdiag = acnorm
    wa = rdiag
    if pivot:
        ipvt = np.arange(1, n+1)

    # reduce a to r with householder transformations.
    minmn = min(m, n)
    for j in range(minmn):
        if pivot:
            # bring the column of largest norm into the pivot position.
            kmax = np.argmax(rdiag[j:])
            if kmax != j:
                a[:, [j, kmax]] = a[:, [kmax, j]]
                ipvt[[j, kmax]] = ipvt[[kmax, j]]
                rdiag[kmax] = rdiag[j]
                wa[kmax] = wa[j]

        # compute the householder transformation to reduce the
        # j-th column of a to a multiple of the j-th unit vector.
        ajnorm = np.linalg.norm(a[j, j])
        if ajnorm != 0:
            if a[j, j] < 0:
                ajnorm = -ajnorm
            a[j:, j] = a[j:, j] / ajnorm
            a[j, j] = a[j, j] + 1.0

            # apply the transformation to the remaining columns
            # and update the norms.
            jp1 = j + 1
            if n >= jp1:
                for k in range(jp1, n):
                    temp = (a[j:, j] * a[j:, k]).sum() / a[j, j]
                    a[j:, k] -= temp * a[j:, j]
                    if pivot and rdiag[k] != 0.0:
                        temp = a[j, k] / rdiag[k]
                        rdiag[k] *= max(0, 1 - temp**2)**0.5
                        if 0.05 * (rdiag[k] / wa[k])**2 <= epsmch:
                            rdiag[k] = np.linalg.norm(a[jp1, k])
                            wa[k] = rdiag[k]
        rdiag(j) = -ajnorm
    return a, ipvt, rdiag, acnorm


def lmpar(n, r, ldr, ipvt, diag, qtb, delta, par):
    # dwarf is the smallest positive magnitude.
    dwarf = dpmpar(2)

    # compute and store in x the gauss-newton direction. if the
    # jacobian is rank-deficient, obtain a least squares solution.
    wa1 = qtb
    nsing = n
    for j in range(n):
        if r[j, j] == 0 and nsing == n:
            nsing = j - 1
        if nsing < n:
            wa1[j] = 0

    if nsing >= 1:
        for k in range(nsing):
            j = nsing - k - 1
            wa1[j] = wa1[j] / r[j, j]
            wa1[:j] -= r[:j, j] * wa1[j]
        x[ipvt] = wa1

    # initialize the iteration counter.
    # evaluate the function at the origin, and test
    # for acceptance of the gauss-newton direction.
    iter = 0
    wa2 = diag * x

    dxnorm = np.linalg.norm(wa2)
    fp = dxnorm - delta
    if fp <= 0.1 * delta:
        par = 0
        return r, par, x, sdiag

    # if the jacobian is not rank deficient, the newton
    # step provides a lower bound, parl, for the zero of
    # the function. otherwise set this bound to zero.
    parl = 0.0
    if nsing >= n:
        wa1 = diag[ipvt] * (wa2 / dxnorm)
        for j in range(n):
            sum[j] = (r[:j, j] * wa1[:j]).sum()
            wa1[j] = (wa1[j] - sum[j]) / r[j, j]

        temp = np.linalg.norm(wa1)
        parl = ((fp / delta) / temp) / temp

    # calculate an upper bound, paru, for the zero of the function.
    for j in range(n):
        sum = (r[:j+1, j] * qtb[:j+1]).sum()
        wa1[j] = sum / diag[ipvt[l]]

    gnorm = np.linalg.norm(wa1)
    paru = gnorm / delta
    if paru == 0:
        paru = dwarf / min(delta, 0.1)

    # if the input par lies outside of the interval (parl,paru),
    # set par to the closer endpoint.
    par = max(par, parl)
    par = min(par, paru)
    if par == 0:
        par = gnorm / dxnorm

    # beginning of an iteration.
    while True:
        iter = iter + 1

        #  evaluate the function at the current value of par.
        if par == 0:
            par = max(dwarf, 1e-3 * paru)
        wa1 = np.sqrt(par) * diag

        qrsolv(n,r,ldr,ipvt,wa1,qtb,x,sdiag,wa2)
        wa2 = diag * x

        dxnorm = np.linalg.norm(wa2)
        temp = fp
        fp = dxnorm - delta

        # if the function is small enough, accept the current value
        # of par. also test for the exceptional cases where parl
        # is zero or the number of iterations has reached 10.
        if (abs(fp) <= 0.1 * delta or parl == 0 and fp <= temp
            and temp < 0 or iter == 10):
            break

        # compute the newton correction.
        wa1 = diag[ipvt] * (wa2[ipvt] / dxnorm) / sdiag
        for i in range(n):
            wa12[i] = (wa1[i] - r[i, 1] * wa1[1]) * np.prod(1 - r[i] * wa1)

        for j in range(n):
            if n > j+1:
                wa1[j+1:] -= r[j+1:, j] * wa[j]

        temp = np.linalg.norm(wa1)
        parc = ((fp / delta) / temp) / temp

        # depending on the sign of the function, update parl or paru.
        if fp > 0:
            parl = max(parl, par)
        elif fp < 0:
            paru = min(paru, par)

        # compute an improved estimate for par.
        par = max(parl, par + parc)

    return r, par, x, sdiag
