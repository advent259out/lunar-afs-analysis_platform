#!/usr/bin/env python3
#
#  Pocket SDR Python AP - GNSS Signal Tracking and NAV data decoding
#
#  Author:
#  T.TAKASU
#
#  History:
#  2021-12-01  1.0  new
#  2022-01-13  1.1  add input from stdout
#                   add options -IQ, -e, -yl, -q
#  2022-01-20  1.2  improve performance
#                   add option -3d
#  2023-12-28  1.3  change receiver channel status format
#  2024-01-12  1.4  support multiple signals
#  2024-02-06  1.5  separate plot functions to sdr_ch_plot.py
#
import sys, math, time, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from sdr_func import *
import sdr_code, sdr_ch, sdr_ch_plot

# constants --------------------------------------------------------------------
CYC_SRCH = 10.0      # signal search cycle (s)
MAX_BUFF = 32        # max number of IF data buffer
NCORR_PLOT = 40      # numober of additional correlators for plot
ESC_COL = '\033[34m' # ANSI escape color blue
ESC_RES = '\033[0m'  # ANSI escape reset
ESC_UCUR = '\033[A'  # ANSI escape cursor up

# disable C/N0 thresholds (acquisition/lost) so all channels are kept
sdr_ch.THRES_CN0 = (float('-inf'), float('-inf'))

# read IF data -----------------------------------------------------------------
def read_data(fp, N, IQ, buff, ix, qsign):
    if fp == None:
        raw = np.frombuffer(sys.stdin.buffer.read(N * IQ), dtype='int8')
    else:
        raw = np.frombuffer(fp.read(N * IQ), dtype='int8')

    if len(raw) < N * IQ:
        return False
    elif IQ == 1: # I
        buff[ix:ix+N] = np.array(raw, dtype='complex64')
    else: # IQ (Q sign inverted in MAX2771)
        # INGM: 20250514 Inverted Q sign for MAX2771
        # buff[ix:ix+N] = np.array(raw[0::2] - raw[1::2] * 1j, dtype='complex64')
        buff[ix:ix+N] = np.array(raw[0::2] + qsign * raw[1::2] * 1j, dtype='complex64')
    return True

# print receiver channel status header -----------------------------------------
def print_head(ch):
    nch = 0
    srch = 0
    for i in range(len(ch)):
        if ch[i].state == 'LOCK':
            nch += 1
        if ch[i].state == 'SRCH':
            srch = i + 1
    print('\r TIME(s):%10.2f%60sSRCH: %3d  LOCK:%3d/%3d' % (ch[0].time, '',
        srch, nch, len(ch)))
    print('%3s %4s %5s %3s %8s %6s %9s %11s %7s %11s %4s %5s %4s %4s %3s' % (
        'CH', 'SAT', 'SIG', 'PRN', 'LOCK(s)', 'C/N0', 'SNR(dB)', 'COFF(ms)',
        'DOP(Hz)', 'ADR(cyc)', 'SYNC', '#NAV', '#ERR', '#LOL', 'NER'))

# receiver channel sync status -------------------------------------------------
def sync_stat(ch):
    return (('S' if ch.trk.sec_sync > 0 else '-') +
        ('B' if ch.nav.ssync > 0 else '-') +
        ('F' if ch.nav.fsync > 0 else '-') +
        ('R' if ch.nav.rev else '-'))

# print receiver channel status ------------------------------------------------
def print_stat(no, ch):
    snr_db = ch.snr if ch.snr == ch.snr else float('-inf')
    print('%s%3d %4s %5s %3d %8.2f %6.1f %9.2f %11.7f %7.1f %11.1f %s %5d %4d %4d %3d%s' % (
        ESC_COL, no, ch.sat, ch.sig, ch.prn, ch.lock * ch.T, ch.cn0,
        snr_db, ch.coff * 1e3, ch.fd, ch.adr, sync_stat(ch),
        ch.nav.count[0], ch.nav.count[1], ch.lost, ch.nav.nerr, ESC_RES))

# update receiver channel status -----------------------------------------------
def update_stat(prns, ch, nrow):
    for i in range(nrow):
        print('%s' % (ESC_UCUR), end='')
    n = 2
    print_head(ch)
    for i in range(len(prns)):
        if ch[i].state == 'LOCK':
            print_stat(i + 1, ch[i])
            n += 1
    for i in range(n, nrow):
        print('%107s' % (''))
        n += 1
    return n

# C/N0 bar ---------------------------------------------------------------------
def cn0_bar(cn0):
    return '|' * np.min([int((cn0 - 30.0) / 1.5), 13])

# show usage -------------------------------------------------------------------
def show_usage():
    print('Usage: pocket_trk.py [-sig sig -prn prn[,...] ...] [-p] [-e] [-toff toff]')
    print('       [-f freq] [-fi freq] [-IQ] [-InQ] [-ti tint] [-ts tspan] [-yl ylim]')
    print('       [-afsp-coh-k K] [-log path] [-q] [file]')
    exit()

#-------------------------------------------------------------------------------
#
#   Synopsis
#
#     pocket_trk.py [-sig sig -prn prn[,...] ...] [-p] [-e] [-toff toff]
#         [-f freq] [-fi freq] [-IQ] [-ti tint] [-ts tspan] [-yl ylim]
#         [-afsp-coh-k K] [-log path] [-q] [file]
#
#   Description
#
#     It tracks GNSS signals in digital IF data and decode navigation data in
#     the signals.
#     If single PRN number by -prn option, it plots correlation power and
#     correlation shape of the specified GNSS signal. If multiple PRN numbers
#     specified by -prn option, it plots C/N0 for each PRN.
#
#   Options ([]: default)
#
#     -sig sig -prn prn[,...] ...
#         A GNSS signal type ID (L1CA, L2CM, ...) and a PRN number list of the
#         signal. For signal type IDs, refer pocket_acq.py manual. The PRN
#         number list shall be PRN numbers or PRN number ranges like 1-32 with
#         the start and the end numbers. They are separated by ",". For
#         GLONASS FDMA signals (G1CA, G2CA), the PRN number is treated as the
#         FCN (frequency channel number). The pair of a signal type ID and a PRN
#         number list can be repeated for multiple GNSS signals to be tracked.
#
#     -p
#         Plot signal tracking status in an integrated window. The window shows
#         correlation envelope, correlation I-Q plot, correlation I/Q to time
#         plot and navigation data decoded. You easily find the signal tracking
#         situation. If multiple PRN number specified in -prn option, only the
#         signal with the first PRN number is plotted. [no plot]
#
#     -e
#         Plot correlation shape as an envelop (SQRT(I^2+Q^2)). [I*sign(IP)]
#
#     -3d
#         3D Plot of correlation shapes. [no]
#
#     -toff toff
#         Time offset from the start of digital IF data in s. [0.0]
#
#     -f freq
#         Sampling frequency of digital IF data in MHz. [12.0]
#
#     -fi freq
#         IF frequency of digital IF data in MHz. The IF frequency is equal 0,
#         the IF data is treated as IQ-sampling without -IQ option (zero-IF).
#         [0.0]
#
#     -IQ
#         IQ-sampling even if the IF frequency is not equal 0.
#
#     -ti tint
#         Update interval of signal tracking status, plot and log in s. [0.1]
#
#     -ts tspan
#         Time span for correlation to time plot in s. [1.0]
#
#     -yl ylim
#         Y-axis limit of plots. [0.3]
#
#     -afsp-coh-k K
#         AFSP 相干累加 K 个码期（默认 1；如 K=5≈10 ms）。
#
#     -log path
#         A Log stream path to write signal tracking status. The log includes
#         decoded navigation data and code offset, including navigation data
#         decoded. The stream path should be one of the followings.
#
#         (1) local file  file path without ':'. The file path can be contain
#             time keywords (%Y, %m, %d, %h, %M) as same as RTKLIB stream.
#         (2) TCP server  :port
#         (3) TCP client  address:port
#
#     -q
#         Suppress showing signal tracking status.
#
#     [file]
#         A file path of the input digital IF data. The format should be a
#         series of int8_t (signed byte) for real-sampling (I-sampling),
#         interleaved int8_t for complex-sampling (IQ-sampling).
#         The Pocket SDR RF-frontend and pocket_dump can be used to capture
#         such digital IF data. If the option omitted, the input is taken
#         from stdin.
#
if __name__ == '__main__':
    sig, plot, env, p3d = 'L1CA', False, False, False
    sigs, prns = [], []
    fs, fi, IQ, toff, tint, tspan, ylim = 12e6, 0.0, 1, 0.0, 0.1, 1.0, 0.3
    ch = {}
    file, log_file, log_lvl, quiet = '', '', 4, 0
    fig = None

    # INGM: 20250514 Set Q sign flag
    inv_q = False
    afsp_coh_k = 1

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-sig':
            i += 1
            sig = sys.argv[i]
        elif sys.argv[i] == '-prn':
            i += 1
            for prn in parse_nums(sys.argv[i]):
                sigs.append(sig)
                prns.append(prn)
        elif sys.argv[i] == '-p':
            plot = True
        elif sys.argv[i] == '-e':
            env = True
        elif sys.argv[i] == '-3d':
            p3d = True
        elif sys.argv[i] == '-toff':
            i += 1
            toff = float(sys.argv[i])
        elif sys.argv[i] == '-f':
            i += 1
            fs = float(sys.argv[i]) * 1e6
        elif sys.argv[i] == '-fi':
            i += 1
            fi = float(sys.argv[i]) * 1e6
        elif sys.argv[i] == '-IQ':
            IQ = 2
        elif sys.argv[i] == '-ti':
            i += 1
            tint = float(sys.argv[i])
        elif sys.argv[i] == '-ts':
            i += 1
            tspan = float(sys.argv[i])
        elif sys.argv[i] == '-yl':
            i += 1
            ylim = float(sys.argv[i])
        elif sys.argv[i] == '-log':
            i += 1
            log_file = sys.argv[i]
        elif sys.argv[i] == '-afsp-coh-k':
            i += 1
            try:
                afsp_coh_k = max(1, min(20, int(float(sys.argv[i]) + 0.5)))
            except:
                afsp_coh_k = 1
        elif sys.argv[i] == '-q':
            quiet = 1
        # INGM: 20250514 Set inverse Q sign flag for MAX2771
        elif sys.argv[i] == '-InQ':
            inv_q = True
        elif sys.argv[i][0] == '-':
            show_usage()
        else:
            file = sys.argv[i];
        i += 1

    T = sdr_code.code_cyc(sig) # code cycle
    if T <= 0.0:
        print('Invalid signal %s.' % (sig))
        exit()

    IQ = 1 if IQ == 1 and fi > 0.0 else 2
    if file == '':
        fp = None
    else:
        try:
            fp = open(file, 'rb')
            fp.seek(int(toff * fs * IQ))
        except:
            print('file open error: %s' % (file))
            exit()

    for i in range(len(prns)):
        ncorr = NCORR_PLOT if plot and i == 0 else 0
        ch[i] = sdr_ch.ch_new(sigs[i], prns[i], fs, fi, add_corr=ncorr)
        ch[i].state = 'SRCH'

    if plot:
        fig = sdr_ch_plot.init(env, p3d, toff, tspan, ylim)

    if log_file != '':
        log_open(log_file)
        log_level(log_lvl)
    # set AFSP coherent accumulation factor for tracking
    try:
        import sdr_ch as _sdr_ch
        _sdr_ch.AFSP_COH_K = afsp_coh_k
    except Exception:
        pass

    N = int(T * fs)
    buff = np.zeros(N * (MAX_BUFF + 1), dtype='complex64')
    ix = 0
    err_stats = {i: [] for i in range(len(prns))}
    snr_hist = {i: [] for i in range(len(prns))}
    nrow = 0
    tt = time.time()
    log(3, '$LOG,%.3f,%s,%d,START FILE=%s FS=%.3f FI=%.3f IQ=%d TOFF=%.3f' %
        (0.0, '', 0, file, fs * 1e-6, fi * 1e-6, IQ, toff))

    try:
        for i in range(0, 1000000000):
            time_rcv = toff + T * (i - 1) # receiver time

            # read IF data to buffer
            # INGM: 20250514 Read data with inverse Q sign
            # if not read_data(fp, N, IQ, buff, N * (i % MAX_BUFF)):
            if not read_data(fp, N, IQ, buff, N * (i % MAX_BUFF), -1 if inv_q else 1):
                break;

            if i == 0:
                continue
            elif i % MAX_BUFF == 0:
                buff[-N:] = buff[:N]

            # update receiver channel
            for j in range(len(ch)):
                sdr_ch.ch_update(ch[j], time_rcv, buff, N * ((i - 1) % MAX_BUFF))
                try:
                    sdr_ch_plot.collect_discriminators(ch[j])
                except Exception:
                    pass
                if ch[j].state == 'LOCK':
                    E = abs(ch[j].trk.C[1])
                    L = abs(ch[j].trk.C[2])
                    if E + L > 0.0:
                        err_code = (E - L) / (E + L) * 0.5 * ch[j].T / len(ch[j].code)
                        err_stats[j].append(err_code)
                    if ch[j].snr == ch[j].snr:
                        snr_hist[j].append((time_rcv, ch[j].snr))

            # update receiver channel state
            if i % int(CYC_SRCH / T) == 0:
                for j in range(len(ch)):
                    ix = (ix + 1) % len(ch)
                    if ch[ix].state == 'IDLE':
                        ch[ix].state = 'SRCH'
                        break

            # update log
            if int(time_rcv * 1000) % 1000 == 0:
                t = datetime.datetime.now(datetime.timezone.utc)
                log(3, '$TIME,%.3f,%d,%d,%d,%d,%d,%.6f,UTC' % (time_rcv, t.year,
                    t.month, t.day, t.hour, t.minute, t.second + t.microsecond
                    * 1e-6))

            if (i - 1) % int(tint / T) != 0:
                continue

            # update receiver channel status
            if not quiet:
                nrow = update_stat(prns, ch, nrow)

            # update plots
            if fig:
                sdr_ch_plot.update(fig, ch[0])
                sdr_ch_plot.title(fig, 'SIG = %s, PRN = %3d, FILE = %s, T = %7.2f s' % (
                    ch[0].sig, ch[0].prn, file, ch[0].time))

            for j in range(len(prns)):
                if ch[j].state != 'LOCK':
                    continue
                log(3, '$CH,%.3f,%s,%d,%d,%.1f,%.1f,%.9f,%.3f,%.3f,%d,%d' %
                    (ch[j].time, ch[j].sig, ch[j].prn, ch[j].lock, ch[j].cn0,
                    ch[j].snr if ch[j].snr == ch[j].snr else float('-inf'),
                    ch[j].coff * 1e3, ch[j].fd, ch[j].adr, ch[j].nav.count[0],
                    ch[j].nav.count[1]))

    except KeyboardInterrupt:
        pass

    tt = time.time() - tt
    log(3, '$LOG,%.3f,%s,%d,END FILE=%s' % (tt, '', 0, file))
    if not quiet:
        print('  TIME(s) = %.3f' % (tt))

    if fp != None:
        fp.close()

    if log_file != '':
        log_close()

    if fig:
        sdr_ch_plot.update(fig)

    try:
        sdr_ch_plot.save_history_plots(total_time=time_rcv)
    except Exception:
        pass

    snr_plot_path = os.path.join(os.path.dirname(__file__), 'snr_vs_time.png')
    snr_plotted = False
    fig_snr, ax_snr = plt.subplots(figsize=(10, 4))
    for j, hist in snr_hist.items():
        if not hist:
            continue
        t = np.array([p[0] for p in hist])
        s = np.array([p[1] for p in hist])
        ax_snr.plot(t, s, lw=1.0, label='CH%-3d %s PRN%-3d' % (j + 1, ch[j].sig, ch[j].prn))
        lin = 10.0 ** (s / 10.0)
        rms_lin = np.sqrt(np.mean(lin ** 2))
        rms_db = 10.0 * np.log10(rms_lin) if rms_lin > 0.0 else float('-inf')
        print('CH%-3d %s PRN%-3d SNR RMS: %.2f dB' % (j + 1, ch[j].sig, ch[j].prn, rms_db))
        snr_plotted = True
    if snr_plotted:
        ax_snr.set_xlabel('Time (s)')
        ax_snr.set_ylabel('SNR (dB)')
        ax_snr.set_title('SNR vs Time')
        ax_snr.grid(True, alpha=0.3)
        if len([1 for hist in snr_hist.values() if hist]) <= 8:
            ax_snr.legend(fontsize=8)
        fig_snr.tight_layout()
        fig_snr.savefig(snr_plot_path, dpi=200)
        plt.close(fig_snr)
        print('Saved SNR plot to %s' % (snr_plot_path))
    else:
        plt.close(fig_snr)

    for j, errs in err_stats.items():
        if not errs:
            continue
        errs = np.array(errs)
        sigma_sec = errs.std()
        sigma_chip = sigma_sec / (ch[j].T / len(ch[j].code))
        sigma_m = sigma_sec * 299792458.0
        print('CH%-3d %s PRN%-3d DLL jitter: %.3e chip (%.3e s, %.3f m)' %
              (j + 1, ch[j].sig, ch[j].prn, sigma_chip, sigma_sec, sigma_m))
