"""
Event Picker v3 — clean layout, working ROI detection.
Run: python event_picker_final.py
Deps: pip install pandas numpy matplotlib scipy
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

try:
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.optimize import curve_fit
    SCIPY = True
except Exception:
    SCIPY = False

# ----------------- helpers -----------------

def linear_interp_time(x1, y1, x2, y2, y_target):
    """Return x where line through (x1,y1)-(x2,y2) hits y_target. Assumes y1!=y2."""
    if y2 == y1:
        return x1
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)

def mono_exp(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def bi_exp(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

def _empty_decay_result():
    return {
        'tau': None, 'tau_fast': None, 'tau_slow': None,
        't_tau': None, 'y_tau': None, 'model': 'none', 'r2': None
    }

def _fallback_tau(t_seg, y_seg, baseline_val, amp, direction, result=None):
    """1/e crossing interpolation fallback."""
    if result is None:
        result = _empty_decay_result()
    if len(t_seg) < 2 or amp == 0:
        result['model'] = '1/e interp'
        return result
    target = baseline_val + amp / np.e
    for j in range(len(t_seg) - 1):
        if direction == 'positive':
            crossed = y_seg[j] >= target >= y_seg[j + 1]
        else:
            crossed = y_seg[j] <= target <= y_seg[j + 1]
        if crossed:
            result['t_tau'] = linear_interp_time(t_seg[j], y_seg[j], t_seg[j + 1], y_seg[j + 1], target)
            result['y_tau'] = target
            result['tau'] = result['t_tau'] - t_seg[0]
            result['model'] = '1/e interp'
            break
    if result['model'] == 'none':
        result['model'] = '1/e interp'
    return result

def fit_decay(t_seg, y_seg, baseline_val, amp, direction):
    """
    Fit mono- and bi-exponential decay models. Return the better R2 fit,
    falling back to 1/e interpolation when scipy fitting is unavailable.
    """
    result = _empty_decay_result()
    if not SCIPY or len(t_seg) < 4 or amp == 0:
        return _fallback_tau(t_seg, y_seg, baseline_val, amp, direction, result)

    t0 = float(t_seg[0])
    t_norm = np.asarray(t_seg, dtype=float) - t0
    y_norm = np.asarray(y_seg, dtype=float) - baseline_val
    dur = float(t_norm[-1] - t_norm[0])
    if dur <= 0:
        return _fallback_tau(t_seg, y_seg, baseline_val, amp, direction, result)

    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    tau0 = max(dur / 3.0, np.finfo(float).eps)
    tau_max = max(dur * 10.0, tau0 * 2.0)

    mono_result = None
    try:
        popt, _ = curve_fit(
            mono_exp, t_norm, y_norm,
            p0=[amp, tau0, 0.0],
            bounds=([-np.inf, np.finfo(float).eps, -np.inf],
                    [np.inf, tau_max, np.inf]),
            maxfev=8000
        )
        y_fit = mono_exp(t_norm, *popt)
        ss_res = float(np.sum((y_norm - y_fit) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mono_result = {'model': 'mono-exp', 'tau': abs(float(popt[1])), 'r2': r2}
    except Exception:
        mono_result = None

    bi_result = None
    try:
        popt, _ = curve_fit(
            bi_exp, t_norm, y_norm,
            p0=[amp * 0.6, max(tau0 * 0.3, np.finfo(float).eps),
                amp * 0.4, max(tau0 * 1.5, np.finfo(float).eps), 0.0],
            bounds=([-np.inf, np.finfo(float).eps, -np.inf, np.finfo(float).eps, -np.inf],
                    [np.inf, tau_max, np.inf, tau_max, np.inf]),
            maxfev=12000
        )
        y_fit = bi_exp(t_norm, *popt)
        ss_res = float(np.sum((y_norm - y_fit) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        taus = sorted([abs(float(popt[1])), abs(float(popt[3]))])
        bi_result = {
            'model': 'bi-exp', 'tau': taus[1],
            'tau_fast': taus[0], 'tau_slow': taus[1], 'r2': r2
        }
    except Exception:
        bi_result = None

    chosen = None
    if mono_result and bi_result:
        chosen = bi_result if bi_result['r2'] > mono_result['r2'] else mono_result
    elif mono_result:
        chosen = mono_result
    elif bi_result:
        chosen = bi_result
    else:
        return _fallback_tau(t_seg, y_seg, baseline_val, amp, direction, result)

    result.update(chosen)
    # Find t_tau via 1/e crossing on the actual signal (use original t_seg for absolute time)
    target = baseline_val + amp / np.e
    t_arr = np.asarray(t_seg, dtype=float)
    y_arr = np.asarray(y_seg, dtype=float)
    for j in range(len(t_arr) - 1):
        if direction == 'positive':
            crossed = y_arr[j] >= target >= y_arr[j + 1]
        else:
            crossed = y_arr[j] <= target <= y_arr[j + 1]
        if crossed:
            result['t_tau'] = linear_interp_time(t_arr[j], y_arr[j], t_arr[j + 1], y_arr[j + 1], target)
            result['y_tau'] = target
            break
    # tau from the exponential fit (popt[1] in normalized time) is already correct —
    # it represents the decay time constant in the same units as t_seg.
    # Do NOT overwrite result['tau'] here; it was set from popt[1] above.
    return result

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _event=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        lbl = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background='#252526',
            foreground='#ffffff',
            relief='flat',
            borderwidth=0,
            font=('Segoe UI', 9),
            wraplength=320,
            padx=8,
            pady=5
        )
        lbl.pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

THEMES = {
    'dark': {
        'window_bg': '#252526',
        'plot_bg': '#2b2b2b',
        'fig_bg': '#2b2b2b',
        'signal': '#88bbee',
        'tick': '#aaaaaa',
        'spine': '#666666',
        'text': '#dddddd',
        'legend_face': '#2b2b2b',
        'legend_edge': '#666666',
        'legend_text': '#dddddd',
        'roi_edge': '#00e5ff',
        'baseline': '#8aa8ff',
        'threshold': '#ffb347',
    }
}

TOOLTIPS = {
    'prominence': 'Minimum peak prominence in signal units. Suggested value is 3x RMS noise.',
    'min_dist': 'Minimum event spacing in samples. Suggested value is 50 ms converted to samples.',
    'hp_cutoff': 'High-pass cutoff in Hz. Requires a known sample rate and scipy.',
    'lp_cutoff': 'Low-pass cutoff in Hz. Must be below Nyquist, sample_rate / 2.',
    'rolling_win': 'Rolling percentile window in samples. Use a window much wider than one event.',
    'rolling_pct': 'Rolling baseline percentile from 0 to 100. Low percentiles fit upward-event baselines.',
    'sample_rate': 'Sample rate in Hz, auto-detected from the median time-column interval.',
    'x_unit': 'Display x-axis as seconds, milliseconds, or samples. Existing ROIs are cleared on change.',
}

# ---------------- draggable line ----------------
class DraggableHLine:
    """Draggable horizontal line on an axes. Click once to place, drag to move."""
    def __init__(self, ax, y, color='gray'):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.y = float(y)
        self.line = ax.axhline(self.y, color=color, linestyle='--', linewidth=1)
        self._press = False
        self.cid_press = self.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        # only start drag if click near the line y
        if abs(event.ydata - self.y) < 0.02 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]):
            self._press = True

    def _on_motion(self, event):
        if not self._press or event.inaxes != self.ax:
            return
        self.y = event.ydata
        self.line.set_ydata([self.y, self.y])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._press:
            return
        self._press = False

    def get_y(self):
        return float(self.y)

    def set_y(self, y):
        self.y = float(y)
        self.line.set_ydata([self.y, self.y])
        self.canvas.draw_idle()

    def remove(self):
        try:
            self.line.remove()
        except Exception:
            pass
        try:
            self.canvas.mpl_disconnect(self.cid_press)
            self.canvas.mpl_disconnect(self.cid_release)
            self.canvas.mpl_disconnect(self.cid_motion)
        except Exception:
            pass

# ---------------- main app ----------------
THEMES = {
    'dark': {
        'window_bg': '#252526',
        'plot_bg': '#2b2b2b',
        'fig_bg': '#2b2b2b',
        'signal': '#88bbee',
        'tick': '#aaaaaa',
        'spine': '#666666',
        'text': '#dddddd',
        'legend_face': '#2b2b2b',
        'legend_edge': '#666666',
        'legend_text': '#dddddd',
        'roi_edge': '#00e5ff',
        'baseline': '#8aa8ff',
        'threshold': '#ffb347',
    }
}

TOOLTIPS = {
    'prominence': 'Minimum peak prominence in signal units. Suggested value is 3x RMS noise.',
    'min_dist': 'Minimum event spacing in samples. Suggested value is 50 ms converted to samples.',
    'hp_cutoff': 'High-pass cutoff in Hz. Requires a known sample rate and scipy.',
    'lp_cutoff': 'Low-pass cutoff in Hz. Must be below Nyquist, sample_rate / 2.',
    'rolling_win': 'Rolling percentile window in samples. Use a window much wider than one event.',
    'rolling_pct': 'Rolling baseline percentile from 0 to 100. Low percentiles fit upward-event baselines.',
    'sample_rate': 'Sample rate in Hz, auto-detected from the median time-column interval.',
    'x_unit': 'Display x-axis as seconds, milliseconds, or samples. Existing ROIs are cleared on change.',
}

# ---------------- draggable line ----------------
class DraggableHLine:
    """Draggable horizontal line on an axes. Click once to place, drag to move."""
    def __init__(self, ax, y, color='gray'):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.y = float(y)
        self.line = ax.axhline(self.y, color=color, linestyle='--', linewidth=1)
        self._press = False
        self.cid_press = self.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        # only start drag if click near the line y
        if abs(event.ydata - self.y) < 0.02 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]):
            self._press = True

    def _on_motion(self, event):
        if not self._press or event.inaxes != self.ax:
            return
        self.y = event.ydata
        self.line.set_ydata([self.y, self.y])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._press:
            return
        self._press = False

    def get_y(self):
        return float(self.y)

    def set_y(self, y):
        self.y = float(y)
        self.line.set_ydata([self.y, self.y])
        self.canvas.draw_idle()

    def remove(self):
        try:
            self.line.remove()
        except Exception:
            pass
        try:
            self.canvas.mpl_disconnect(self.cid_press)
            self.canvas.mpl_disconnect(self.cid_release)
            self.canvas.mpl_disconnect(self.cid_motion)
        except Exception:
            pass

# ---------------- main app ----------------

# ═══════════════════════════════════════════════
#  DraggableHLine  (unchanged — proven working)
# ═══════════════════════════════════════════════
class DraggableHLine:
    def __init__(self, ax, y, color='gray'):
        self.ax = ax; self.canvas = ax.figure.canvas
        self.y = float(y)
        self.line = ax.axhline(self.y, color=color, linestyle='--', linewidth=1.2)
        self._press = False
        self.cid_press   = self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion  = self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
    def _on_press(self, e):
        if e.inaxes != self.ax or e.ydata is None: return
        span = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        if abs(e.ydata - self.y) < 0.02 * abs(span): self._press = True
    def _on_motion(self, e):
        if not self._press or e.inaxes != self.ax or e.ydata is None: return
        self.y = e.ydata; self.line.set_ydata([self.y, self.y]); self.canvas.draw_idle()
    def _on_release(self, _): self._press = False
    def get_y(self): return float(self.y)
    def set_y(self, y): self.y=float(y); self.line.set_ydata([self.y,self.y]); self.canvas.draw_idle()
    def remove(self):
        try: self.line.remove()
        except: pass
        for cid in (self.cid_press, self.cid_release, self.cid_motion):
            try: self.canvas.mpl_disconnect(cid)
            except: pass

# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════
class EventPickerApp:
    def __init__(self, master):
        self.master = master
        master.overrideredirect(True)
        master.geometry('1300x900')
        master.minsize(900, 600)
        master.configure(bg='#252526')
        master.option_add('*highlightThickness', 0)
        master.option_add('*BorderWidth', 0)

        
        # Custom gray dark title bar
        self.title_bar = tk.Frame(master, bg='#2b2b2b', relief='flat', bd=0, height=32)
        self.title_bar.pack(side=tk.TOP, fill=tk.X)

        self.title_label = tk.Label(
            self.title_bar,
            text='  Event Picker DhS',
            bg='#2b2b2b',
            fg='#dddddd',
            font=('Segoe UI', 10)
        )
        self.title_label.pack(side=tk.LEFT, pady=6)

        btn_frame = tk.Frame(self.title_bar, bg='#2b2b2b')
        btn_frame.pack(side=tk.RIGHT)

        self.min_btn = tk.Button(
            btn_frame,
            text='—',
            bg='#2b2b2b',
            fg='#dddddd',
            activebackground='#3a3a3a',
            activeforeground='white',
            relief='flat',
            bd=0,
            width=4,
            command=lambda: master.iconify()
        )
        self.min_btn.pack(side=tk.LEFT)

        self.maximized = False

        def toggle_maximize():
            if self.maximized:
                master.state('normal')
                self.maximized = False
                self.max_btn.config(text='□')
            else:
                master.state('zoomed')
                self.maximized = True
                self.max_btn.config(text='❐')

        self.max_btn = tk.Button(
            btn_frame,
            text='□',
            bg='#2b2b2b',
            fg='#dddddd',
            activebackground='#3a3a3a',
            activeforeground='white',
            relief='flat',
            bd=0,
            width=4,
            command=toggle_maximize
        )
        self.max_btn.pack(side=tk.LEFT)

        self.close_btn = tk.Button(
            btn_frame,
            text='✕',
            bg='#2b2b2b',
            fg='#dddddd',
            activebackground='#c42b1c',
            activeforeground='white',
            relief='flat',
            bd=0,
            width=4,
            command=master.destroy
        )
        self.close_btn.pack(side=tk.LEFT)

        def start_move(event):
            master.x = event.x
            master.y = event.y

        def do_move(event):
            if not self.maximized:
                x = event.x_root - master.x
                y = event.y_root - master.y
                master.geometry(f'+{x}+{y}')

        self.title_bar.bind('<Button-1>', start_move)
        self.title_bar.bind('<B1-Motion>', do_move)
        self.title_label.bind('<Button-1>', start_move)
        self.title_label.bind('<B1-Motion>', do_move)

        # Remove any highlight border around the window content area
        try:
            master.tk.call('tk', 'windowingsystem')  # just a no-op probe
            master.option_add('*highlightThickness', 0)
            master.option_add('*highlightBackground', '#252526')
            master.option_add('*highlightColor', '#252526')
            master.option_add('*Background', '#252526')
            master.option_add('*Foreground', '#dddddd')
            master.option_add('*activeBackground', '#2a2a2a')
            master.option_add('*activeForeground', '#ffffff')
        except Exception:
            pass

        # ── data ──
        self.t = self.y = self.raw_y = self.raw_t = None
        self.fs = None; self.fs_manual = False
        self.x_unit = 'seconds'

        # ── state ──
        self.h_baseline = self.h_threshold = None
        self.rois = []; self.peaks = []
        self.peak_direction = 'positive'
        self.theme_name = 'dark'
        self.rolling_active = False
        self.baseline_correction = None
        self.min_prominence = 0.0
        self.min_distance = 1
        self.click_mode = None; self.cid_click = None
        self.rect = None; self.rect_selector = None

        # ── build UI then init ──
        self._build_ui()
        self._apply_theme()
        self._start_rect_selector()

    # ───────────────────────────────────────────
    #  UI BUILD  — grid-based, 3 rows of controls
    # ───────────────────────────────────────────
    def _build_ui(self):
        m = self.master

        # ── outer structure ──
        # controls_frame: fixed height at top
        # canvas_frame:   expands to fill middle
        # status_frame:   fixed height at bottom

        DBG = '#252526'  # dark bg for all frames
        ctrl = tk.Frame(m, bd=0, bg=DBG)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        # plot area frame
        plot_frame = tk.Frame(m, bg=DBG, bd=0)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # bottom strip: toolbar | buttons | log — packed BOTTOM before canvas
        bot_frame = tk.Frame(m, bg=DBG, bd=0)
        bot_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # ── 4 control rows inside ctrl ──
        r1 = tk.Frame(ctrl, bg=DBG, bd=0); r1.pack(fill=tk.X, pady=1)
        r2 = tk.Frame(ctrl, bg=DBG, bd=0); r2.pack(fill=tk.X, pady=1)
        r3 = tk.Frame(ctrl, bg=DBG, bd=0); r3.pack(fill=tk.X, pady=1)
        r4 = tk.Frame(ctrl, bg=DBG, bd=0); r4.pack(fill=tk.X, pady=1)

        P = dict(padx=4, pady=2)  # button padding

        # ── ROW 1: file, xunit, Fs, theme ──
        tk.Button(r1, text='Load CSV/Excel', command=self.load_file, **{}).pack(side=tk.LEFT, **P)
        tk.Label(r1,  text='X-unit:').pack(side=tk.LEFT, padx=(8,2))
        self.xunit_var = tk.StringVar(value='seconds')
        xc = ttk.Combobox(r1, textvariable=self.xunit_var,
                          values=['seconds','ms','samples'], width=8, state='readonly')
        xc.pack(side=tk.LEFT, padx=2)
        xc.bind('<<ComboboxSelected>>', self._on_xunit_change)
        tk.Label(r1, text='Fs (Hz):').pack(side=tk.LEFT, padx=(10,2))
        self.fs_entry = tk.Entry(r1, width=8); self.fs_entry.pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text='Set Fs', command=self._override_fs).pack(side=tk.LEFT, **P)
        self.fs_status = tk.Label(r1, text='Fs: —', width=18, anchor='w')
        self.fs_status.pack(side=tk.LEFT, padx=4)

        # ── ROW 2: baseline/threshold/ROI controls ──
        tk.Button(r2, text='Place Baseline',  command=self.place_baseline_mode).pack(side=tk.LEFT, **P)
        tk.Button(r2, text='Place Threshold', command=self.place_threshold_mode).pack(side=tk.LEFT, **P)
        tk.Label(r2,  text='  Peak:').pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value='positive')
        tk.Radiobutton(r2, text='Positive', variable=self.dir_var, value='positive', command=self.set_dir).pack(side=tk.LEFT)
        tk.Radiobutton(r2, text='Negative', variable=self.dir_var, value='negative', command=self.set_dir).pack(side=tk.LEFT)
        tk.Label(r2, text='  Prominence:').pack(side=tk.LEFT, padx=(8,2))
        self.prom_entry = tk.Entry(r2, width=8); self.prom_entry.insert(0,'0'); self.prom_entry.pack(side=tk.LEFT)
        ToolTip(self.prom_entry, TOOLTIPS['prominence'])
        tk.Label(r2, text='  Min dist(samples):').pack(side=tk.LEFT, padx=(6,2))
        self.dist_entry = tk.Entry(r2, width=6); self.dist_entry.insert(0,'1'); self.dist_entry.pack(side=tk.LEFT)
        ToolTip(self.dist_entry, TOOLTIPS['min_dist'])
        tk.Button(r2, text='Apply params', command=self.apply_params).pack(side=tk.LEFT, **P)
        tk.Label(r2, text='  ').pack(side=tk.LEFT)
        tk.Button(r2, text='Clear ROIs',    command=self.clear_rois).pack(side=tk.LEFT, **P)
        tk.Button(r2, text='Undo ROI',      command=self.undo_last_roi).pack(side=tk.LEFT, **P)
        tk.Button(r2, text='Delete ROI #',  command=self.delete_roi_by_index).pack(side=tk.LEFT, **P)
        tk.Button(r2, text='Clear peaks in ROI', command=self.remove_peaks_in_last_roi).pack(side=tk.LEFT, **P)

        # ── ROW 3: filters ──
        tk.Label(r3, text='HP (Hz):').pack(side=tk.LEFT, padx=(4,2))
        self.hp_entry = tk.Entry(r3, width=7); self.hp_entry.pack(side=tk.LEFT)
        ToolTip(self.hp_entry, TOOLTIPS['hp_cutoff'])
        tk.Button(r3, text='Apply HP', command=self.apply_hp).pack(side=tk.LEFT, **P)
        tk.Label(r3, text='  LP (Hz):').pack(side=tk.LEFT, padx=(6,2))
        self.lp_entry = tk.Entry(r3, width=7); self.lp_entry.pack(side=tk.LEFT)
        ToolTip(self.lp_entry, TOOLTIPS['lp_cutoff'])
        tk.Button(r3, text='Apply LP',  command=self.apply_lp).pack(side=tk.LEFT, **P)
        tk.Button(r3, text='Reset Raw', command=self.reset_raw).pack(side=tk.LEFT, **P)
        if not SCIPY:
            tk.Label(r3, text='  ⚠ scipy not found', fg='#ff8844').pack(side=tk.LEFT)

        # ── ROW 4: rolling baseline ──
        self.rolling_var = tk.BooleanVar(value=False)
        tk.Checkbutton(r4, text='Rolling Baseline', variable=self.rolling_var,
                       command=self._toggle_rolling).pack(side=tk.LEFT, padx=4)
        tk.Label(r4, text='Window (samples):').pack(side=tk.LEFT, padx=(6,2))
        self.roll_win_entry = tk.Entry(r4, width=7); self.roll_win_entry.insert(0,'20')
        self.roll_win_entry.pack(side=tk.LEFT)
        ToolTip(self.roll_win_entry, TOOLTIPS['rolling_win'])
        tk.Label(r4, text='  Percentile:').pack(side=tk.LEFT, padx=(6,2))
        self.roll_pct_entry = tk.Entry(r4, width=5); self.roll_pct_entry.insert(0,'10')
        self.roll_pct_entry.pack(side=tk.LEFT)
        ToolTip(self.roll_pct_entry, TOOLTIPS['rolling_pct'])
        tk.Button(r4, text='Apply Rolling', command=self._apply_rolling).pack(side=tk.LEFT, **P)

        # ── MATPLOTLIB FIGURE ──
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.09)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── BOTTOM: log | toolbar | export buttons ──
        # log (bottom-most)
        self.info = tk.Text(bot_frame, height=6, font=('Courier', 9), bg='#303030', fg='#88cc88', relief='flat', bd=0, highlightthickness=0)
        self.info.pack(side=tk.BOTTOM, fill=tk.X)

        # toolbar + export on same row
        action_row = tk.Frame(bot_frame, bg=DBG, bd=0)
        action_row.pack(side=tk.BOTTOM, fill=tk.X)

        export_frame = tk.Frame(action_row, bg=DBG, bd=0)
        export_frame.pack(side=tk.RIGHT, padx=4, pady=2)
        tk.Button(export_frame, text='Export CSV',     command=self.export_events).pack(side=tk.LEFT, **P)
        tk.Button(export_frame, text='Save Figure',    command=self.save_figure).pack(side=tk.LEFT, **P)
        tk.Button(export_frame, text='Summary Stats',  command=self.show_summary).pack(side=tk.LEFT, **P)

        tb_frame = tk.Frame(action_row, bg=DBG, bd=0)
        tb_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Custom dark toolbar
        toolbar_row = tk.Frame(tb_frame, bg='#252526')
        toolbar_row.pack(side=tk.LEFT, pady=2)

        def tb_btn(txt, cmd):
            return tk.Button(
                toolbar_row,
                text=txt,
                command=cmd,
                bg='#303030',
                fg='#dddddd',
                activebackground='#404040',
                activeforeground='white',
                relief='flat',
                bd=0,
                padx=10,
                pady=4,
                font=('Segoe UI Symbol', 11)
            )

        
        b1 = tb_btn('⌂', lambda: self.canvas.toolbar.home())
        b1.pack(side=tk.LEFT, padx=2)
        ToolTip(b1, 'Reset view')

        b2 = tb_btn('←', lambda: self.canvas.toolbar.back())
        b2.pack(side=tk.LEFT, padx=2)
        ToolTip(b2, 'Previous view')

        b3 = tb_btn('→', lambda: self.canvas.toolbar.forward())
        b3.pack(side=tk.LEFT, padx=2)
        ToolTip(b3, 'Next view')

        b4 = tb_btn('✥', lambda: self.canvas.toolbar.pan())
        b4.pack(side=tk.LEFT, padx=2)
        ToolTip(b4, 'Pan tool')

        b5 = tb_btn('⌕', lambda: self.canvas.toolbar.zoom())
        b5.pack(side=tk.LEFT, padx=2)
        ToolTip(b5, 'Zoom tool')

        b6 = tb_btn('💾', lambda: self.canvas.toolbar.save_figure())
        b6.pack(side=tk.LEFT, padx=2)
        ToolTip(b6, 'Save figure')


        
        # init log
        self.info.insert(tk.END, 'Load a CSV/Excel file to begin.\n')
        self.info.insert(tk.END, 'ZOOM: magnifier icon in toolbar. HOME icon resets. ROI drag works at any zoom.\n')


    # ---------- file/load ----------
    def load_file(self):
        fname = filedialog.askopenfilename(filetypes=[('CSV/Excel','*.csv *.xls *.xlsx'),('All files','*.*')])
        if not fname:
            return
        try:
            if fname.lower().endswith('.csv'):
                df = pd.read_csv(fname)
            else:
                df = pd.read_excel(fname)
        except Exception as e:
            messagebox.showerror('Error','Could not read file: '+str(e))
            return
        if df.shape[1] < 2:
            messagebox.showerror('Error','File must have at least two columns')
            return
        t_raw = np.array(df.iloc[:,0],dtype=float)
        y_raw = np.array(df.iloc[:,1],dtype=float)
        self.fname = fname
        self.raw_t = t_raw.copy()
        self.x_unit = self.xunit_var.get()
        self.t = self._convert_t(self.raw_t)
        self.y = y_raw.copy()
        self.raw_y = self.y.copy()
        self.baseline_correction = None
        self.rolling_active = False
        self.rolling_var.set(False)
        self.fs_manual = False
        self._auto_detect_fs()
        self._suggest_params()
        # defaults
        self.baseline = float(np.median(self.y[:max(1,int(0.05*len(self.y)))])*1.0)
        self.threshold = self.baseline + 0.5*np.std(self.y)
        # create draggable lines
        if self.h_baseline:
            self.h_baseline.remove()
        self.h_baseline = DraggableHLine(self.ax, self.baseline, color=THEMES[self.theme_name]['baseline'])
        if self.h_threshold:
            self.h_threshold.remove()
        self.h_threshold = DraggableHLine(self.ax, self.threshold, color=THEMES[self.theme_name]['threshold'])
        self.rois = []
        self.peaks = []
        self.redraw()
        self.info.insert(tk.END, f'Loaded {fname} -- {len(self.t)} samples.\n')
        if self.fs:
            self.info.insert(tk.END, f'Auto-detected Fs={self.fs:.6g} Hz from time column median interval.\n')

    def _convert_t(self, t_raw):
        unit = self.xunit_var.get() if hasattr(self, 'xunit_var') else self.x_unit
        if unit == 'ms':
            return np.asarray(t_raw, dtype=float) * 1000.0
        if unit == 'samples':
            return np.arange(len(t_raw), dtype=float)
        return np.asarray(t_raw, dtype=float).copy()

    def _auto_detect_fs(self):
        self.fs = None
        if self.raw_t is None or len(self.raw_t) < 2:
            self._update_fs_status()
            return
        diffs = np.diff(self.raw_t)
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs > 0]
        if len(diffs):
            dt = float(np.median(diffs))
            if dt > 0:
                self.fs = 1.0 / dt
        self._update_fs_status()

    def _update_fs_status(self):
        if not hasattr(self, 'fs_status'):
            return
        if self.fs and self.fs > 0:
            suffix = 'manual' if self.fs_manual else 'auto'
            self.fs_status.configure(text=f'Fs: {self.fs:.6g} Hz ({suffix})')
            self.fs_entry.delete(0, tk.END)
            self.fs_entry.insert(0, f'{self.fs:.6g}')
        else:
            self.fs_status.configure(text='Fs: unknown')

    def _suggest_params(self):
        if self.y is None or len(self.y) < 2:
            return
        diffs = np.diff(self.y)
        rms_noise = float(np.std(diffs) / np.sqrt(2.0)) if len(diffs) else float(np.std(self.y))
        prom = 3.0 * rms_noise
        if self.fs and self.fs > 0:
            min_dist = max(1, int(round(0.050 * self.fs)))
        else:
            min_dist = 1
        roll_win = max(20 * min_dist, int(round(0.10 * len(self.y))))
        roll_win = max(1, roll_win)

        self.min_prominence = prom
        self.min_distance = min_dist
        self.prom_entry.delete(0, tk.END)
        self.prom_entry.insert(0, f'{prom:.6g}')
        self.dist_entry.delete(0, tk.END)
        self.dist_entry.insert(0, str(min_dist))
        self.roll_win_entry.delete(0, tk.END)
        self.roll_win_entry.insert(0, str(roll_win))
        self.info.insert(
            tk.END,
            f'Suggested params: prominence={prom:.6g}, min_dist={min_dist} samples, '
            f'rolling_window={roll_win} samples.\n'
        )

    def _override_fs(self):
        try:
            fs = float(self.fs_entry.get())
            if fs <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror('Input', 'Enter a positive sample rate in Hz.')
            return
        self.fs = fs
        self.fs_manual = True
        self._update_fs_status()
        self._suggest_params()
        self.info.insert(tk.END, f'Sample rate manually set to {fs:.6g} Hz.\n')

    def _on_xunit_change(self, _event=None):
        if self.raw_t is None:
            self.x_unit = self.xunit_var.get()
            self.redraw()
            return
        self.x_unit = self.xunit_var.get()
        self.t = self._convert_t(self.raw_t)
        self.rois = []
        self.peaks = []
        self._suggest_params()
        self.redraw()
        self.info.insert(tk.END, f'X-axis unit changed to {self.x_unit}; cleared ROIs and peaks.\n')

    def _get_fs_hz(self):
        return self.fs if self.fs and self.fs > 0 else None

    def _toggle_theme(self):
        return

    def _apply_theme(self):
        C = THEMES[self.theme_name]
        bg      = C['window_bg']
        fg      = C['text']
        is_dark = True
        btn_bg  = '#2b2b2b'
        btn_fg  = '#dddddd'
        ent_bg  = '#1f1f1f'
        ent_fg  = '#dddddd'
        sel_bg  = '#444444'

        # --- ttk style (covers Combobox) ---
        style = ttk.Style()
        try:
            style.theme_use('default')
        except Exception:
            pass
        style.configure('TCombobox',
                        fieldbackground=ent_bg, background=btn_bg,
                        foreground=ent_fg, selectbackground=sel_bg,
                        selectforeground=ent_fg, arrowcolor=fg)
        style.map('TCombobox',
                  fieldbackground=[('readonly', ent_bg)],
                  foreground=[('readonly', ent_fg)],
                  background=[('readonly', btn_bg)])
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg)

        # --- recursive tk widget walker ---
        def _style_all(widget):
            cls = widget.winfo_class()
            try:
                if cls in ('Frame', 'Labelframe', 'Toplevel'):
                    widget.configure(bg=bg)
                elif cls == 'Button':
                    widget.configure(
                        bg='#303030',
                        fg='#dddddd',
                        activebackground='#2b2b2b',
                        activeforeground='#ffffff',
                        relief='flat',
                        bd=0,
                        highlightthickness=0,
                        padx=8,
                        pady=4
                    )
                elif cls == 'Label':
                    widget.configure(bg=bg, fg=fg)
                elif cls == 'Entry':
                    widget.configure(bg=ent_bg, fg=ent_fg,
                                     insertbackground=fg, relief='flat',
                                     bd=1, highlightthickness=0)
                elif cls == 'Checkbutton':
                    widget.configure(bg=bg, fg=fg, selectcolor=ent_bg,
                                     activebackground=bg, activeforeground=fg,
                                     highlightthickness=0)
                elif cls == 'Radiobutton':
                    widget.configure(bg=bg, fg=fg, selectcolor=ent_bg,
                                     activebackground=bg, activeforeground=fg,
                                     highlightthickness=0)
                elif cls == 'Text':
                    pass  # handled separately below
            except Exception:
                pass
            for child in widget.winfo_children():
                _style_all(child)

        try:
            self.master.configure(bg=bg)
        except Exception:
            pass
        _style_all(self.master)

        # Style toolbar recursively — covers back/forward nav buttons too
        if hasattr(self, '_toolbar'):
            def _style_toolbar_widget(w):
                try:
                    cls = w.winfo_class()
                    if cls in ('Frame', 'Labelframe'):
                        w.configure(bg=bg)
                    
                    elif cls == 'Button':
                        w.configure(
                            bg='#303030',
                            fg='#dddddd',
                            activebackground='#404040',
                            activeforeground='#ffffff',
                            relief='flat',
                            bd=0,
                            highlightthickness=0,
                            padx=4,
                            pady=2
                        )

                    elif cls == 'Label':
                        w.configure(bg=bg, fg=fg)
                    elif cls == 'Entry':
                        w.configure(bg=ent_bg, fg=ent_fg,
                                    insertbackground=fg, relief='flat',
                                    bd=1, highlightthickness=0)
                    elif cls == 'Canvas':
                        w.configure(bg=bg, highlightthickness=0)
                except Exception:
                    pass
                for child in w.winfo_children():
                    _style_toolbar_widget(child)
            try:
                self._toolbar.configure(bg='#252526', relief='flat', bd=0, highlightthickness=0)
            except Exception:
                pass
            _style_toolbar_widget(self._toolbar)
        if hasattr(self, 'info'):
            self.info.configure(
                bg='#303030',
                fg='#88cc88',
                insertbackground=fg)
        if hasattr(self, 'fig') and hasattr(self, 'ax'):
            self.fig.patch.set_facecolor(C['fig_bg'])
            self.ax.set_facecolor(C['plot_bg'])
            self.ax.tick_params(colors=C['tick'])
            for spine in self.ax.spines.values():
                spine.set_edgecolor(C['spine'])
            self.redraw()

    def _toggle_rolling(self):
        if self.rolling_var.get():
            self._apply_rolling()
            return
        if self.baseline_correction is not None:
            self.y = self.y + self.baseline_correction
            self.baseline_correction = None
        self.rolling_active = False
        self.redraw()
        self.info.insert(tk.END, 'Rolling baseline correction removed.\n')

    def _apply_rolling(self):
        if self.y is None:
            return
        try:
            win_samples = int(float(self.roll_win_entry.get()))
            pct = float(self.roll_pct_entry.get())
            if win_samples < 3 or not (0 <= pct <= 100):
                raise ValueError
        except Exception:
            messagebox.showerror('Input', 'Enter rolling window >= 3 samples and percentile from 0 to 100.')
            self.rolling_var.set(False)
            return
        if self.baseline_correction is not None:
            self.y = self.y + self.baseline_correction
        rolling = (
            pd.Series(self.y)
            .rolling(window=win_samples, center=True, min_periods=1)
            .quantile(pct / 100.0)
            .values
        )
        self.baseline_correction = rolling
        self.y = self.y - rolling
        self.rolling_active = True
        self.rolling_var.set(True)
        self.redraw()
        self.info.insert(
            tk.END,
            f'Rolling baseline applied: window={win_samples} samples, percentile={pct:.4g}%.\n'
        )

    # ---------- UI small helpers ----------
    def set_dir(self):
        self.peak_direction = self.dir_var.get()
        self.info.insert(tk.END, f'Peak direction: {self.peak_direction}\n')

    def apply_params(self):
        try:
            self.min_prominence = float(self.prom_entry.get())
        except Exception:
            self.min_prominence = 0.0
        try:
            self.min_distance = max(1, int(float(self.dist_entry.get())))
        except Exception:
            self.min_distance = 1
        self.info.insert(tk.END, f'Params: prominence={self.min_prominence}, min_dist={self.min_distance}\n')

    def place_baseline_mode(self):
        self.click_mode = 'baseline'
        self._enable_click()
        self.info.insert(tk.END, 'Click on plot to place baseline (then drag).\n')

    def place_threshold_mode(self):
        self.click_mode = 'threshold'
        self._enable_click()
        self.info.insert(tk.END, 'Click on plot to place threshold (then drag).\n')

    def _enable_click(self):
        if self.cid_click:
            self.canvas.mpl_disconnect(self.cid_click)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.click_mode == 'baseline':
            y = event.ydata
            if self.h_baseline:
                self.h_baseline.remove()
            self.h_baseline = DraggableHLine(self.ax, y, color=THEMES[self.theme_name]['baseline'])
            self.click_mode = None
            self.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
            self.redraw()
        elif self.click_mode == 'threshold':
            y = event.ydata
            if self.h_threshold:
                self.h_threshold.remove()
            self.h_threshold = DraggableHLine(self.ax, y, color=THEMES[self.theme_name]['threshold'])
            self.click_mode = None
            self.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
            self.redraw()

    # ---------- ROI handling ----------
    def clear_rois(self):
        self.rois = []
        self.peaks = []
        self.info.insert(tk.END, 'Cleared ROIs and peaks.\n')
        self.redraw()

    def _start_rect_selector(self):
        
        """Create (or reactivate) a RectangleSelector and ensure it's active."""
        try:
            mode = self.canvas.toolbar.mode
        except Exception:
            mode = None
        if mode and mode != '':
            self.info.insert(tk.END, f'WARNING: toolbar mode is \"{mode}\" — disable pan/zoom first.\n')

        if self.rect_selector is not None:
            try:
                self.rect_selector.set_active(True)
                self.info.insert(tk.END, 'ROI selector activated (reused). Drag to draw ROI.\n')
                return
            except Exception:
                self.rect_selector = None

        try:
            self.rect_selector = RectangleSelector(self.ax, self._on_select,
                                                drawtype='box', useblit=True,
                                                button=[1], minspanx=5, minspany=5, spancoords='data')
        except TypeError:
            self.rect_selector = RectangleSelector(self.ax, self._on_select)

        try:
            self.rect_selector.set_active(True)
        except Exception:
            pass

        self.canvas.draw_idle()
        self.info.insert(tk.END, 'ROI selector created and active — drag to draw.\n')

    def _on_select(self, eclick, erelease):
    
        """Robust ROI callback with full debug logging."""
        try:
            x0, y0 = float(eclick.xdata), float(eclick.ydata)
            x1, y1 = float(erelease.xdata), float(erelease.ydata)
        except Exception:
            self.info.insert(tk.END, 'ROI cancelled: release outside axes.\n')
            return

        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))

        self.info.insert(tk.END, f'_on_select fired: x[{xa:.4g},{xb:.4g}] y[{ya:.4g},{yb:.4g}]\n')

        if xb <= xa or yb <= ya:
            self.info.insert(tk.END, 'ROI ignored: zero area.\n')
            return

        roi = (xa, xb, ya, yb)
        self.rois.append(roi)

        try:
            self.detect_roi(roi)
        except Exception as e:
            self.info.insert(tk.END, f'detect_roi ERROR: {e}\n')

        self.redraw()
        self.canvas.draw_idle()

    def _deactivate_rect_selector(self):
        if self.rect_selector is None:
            self.info.insert(tk.END, 'ROI selector not active.\n')
            return
        try:
            self.rect_selector.set_active(False)
        except Exception:
            pass
        self.info.insert(tk.END, 'ROI selector deactivated.\n')


    def detect_roi(self, roi):
    
        """Detect exactly one (best) peak in the ROI and append it to self.peaks."""
        if self.t is None:
            return
        x0, x1, y0, y1 = roi
        # baseline & threshold values
        baseline_val = self.h_baseline.get_y() if self.h_baseline else float(np.median(self.y))
        threshold_val = self.h_threshold.get_y() if self.h_threshold else baseline_val + 0.5*np.std(self.y)

        idx = np.where((self.t >= x0) & (self.t <= x1))[0]
        if len(idx) < 3:
            self.info.insert(tk.END, 'ROI too small for detection.\n')
            return
        ti = self.t[idx]
        yi = self.y[idx]

        # detection: get all candidate peak indices relative to yi
        candidates = []
        if SCIPY:
            if self.peak_direction == 'positive':
                peaks, props = find_peaks(yi, prominence=self.min_prominence, distance=self.min_distance)
            else:
                peaks, props = find_peaks(-yi, prominence=self.min_prominence, distance=self.min_distance)
            candidates = list(peaks)
        else:
            if self.peak_direction == 'positive':
                candidates = [i for i in range(1,len(yi)-1) if (yi[i]>yi[i-1] and yi[i]>=yi[i+1] and yi[i]>=threshold_val)]
            else:
                candidates = [i for i in range(1,len(yi)-1) if (yi[i]<yi[i-1] and yi[i]<=yi[i+1] and yi[i]<=threshold_val)]

        if not candidates:
            self.info.insert(tk.END, f'No candidate peaks found in ROI x[{x0:.4g},{x1:.4g}].\n')
            return

        # Choose the single best candidate: largest absolute amplitude relative to baseline
        best_c = None
        best_amp = -np.inf
        for c in candidates:
            gi = idx[c]
            amp = abs(self.y[gi] - baseline_val)
            if amp > best_amp:
                best_amp = amp
                best_c = c

        # If somehow none chosen, bail
        if best_c is None:
            return

        # Convert local index to global index
        gi = idx[best_c]
        tpk = float(self.t[gi])
        ypk = float(self.y[gi])

        # onset: walk left until crossing baseline (interpolate)
        onset_idx = gi
        while onset_idx>0 and ((self.y[onset_idx]-baseline_val)*(ypk-baseline_val) > 0):
            onset_idx -= 1
        if onset_idx < gi:
            x1t, y1v = self.t[onset_idx], self.y[onset_idx]
            x2t, y2v = self.t[onset_idx+1], self.y[onset_idx+1]
            t_on = linear_interp_time(x1t,y1v,x2t,y2v, baseline_val)
            y_on = baseline_val
        else:
            t_on = float(self.t[onset_idx])
            y_on = float(self.y[onset_idx])

        # end: walk right until signal stays below baseline for CONSEC_END consecutive
        # samples. This prevents a single noise dip from prematurely truncating the event.
        CONSEC_END = max(2, int(round((self.fs or 10) * 0.1)))  # ~100 ms
        end_idx = gi
        j = gi
        consec_below = 0
        first_below = None
        while j < len(self.y) - 1:
            if (self.y[j] - baseline_val) * (ypk - baseline_val) <= 0:
                if first_below is None:
                    first_below = j
                consec_below += 1
                if consec_below >= CONSEC_END:
                    end_idx = first_below
                    break
            else:
                consec_below = 0
                first_below = None
                end_idx = j
            j += 1
        else:
            end_idx = len(self.y) - 1

        if end_idx > gi:
            x1t, y1v = self.t[end_idx - 1], self.y[end_idx - 1]
            x2t, y2v = self.t[end_idx], self.y[end_idx]
            t_end = linear_interp_time(x1t, y1v, x2t, y2v, baseline_val)
            y_end = baseline_val
        else:
            t_end = float(self.t[end_idx])
            y_end = float(self.y[end_idx])

        amp = ypk - baseline_val

        # rise 10-90% (interpolated crossings from onset to peak)
        rt10 = rt90 = None
        if amp != 0:
            y10 = baseline_val + 0.1 * amp
            y90 = baseline_val + 0.9 * amp
            for j in range(onset_idx + 1, gi + 1):
                if j < len(self.y):
                    crossed10 = (
                        (self.peak_direction == 'positive' and self.y[j - 1] < y10 <= self.y[j]) or
                        (self.peak_direction == 'negative' and self.y[j - 1] > y10 >= self.y[j])
                    )
                    if crossed10:
                        rt10 = linear_interp_time(self.t[j-1], self.y[j-1], self.t[j], self.y[j], y10)
                        break
            for j in range(onset_idx + 1, gi + 1):
                if j < len(self.y):
                    crossed90 = (
                        (self.peak_direction == 'positive' and self.y[j - 1] < y90 <= self.y[j]) or
                        (self.peak_direction == 'negative' and self.y[j - 1] > y90 >= self.y[j])
                    )
                    if crossed90:
                        rt90 = linear_interp_time(self.t[j-1], self.y[j-1], self.t[j], self.y[j], y90)
                        break
        rise_time = (rt90 - rt10) if (rt10 is not None and rt90 is not None) else None

        # decay tau (1/e) — find crossing via consecutive-sample interpolation (no j==gi snap)
        t_tau = y_tau = None
        if amp != 0:
            target = baseline_val + amp / np.e
            # Search from peak+1 to a generous window (3x past end_idx for long tails)
            search_end = min(len(self.y), end_idx + max(int(self.fs or 10) * 3, 10))
            for j in range(gi + 1, search_end):
                if self.peak_direction == 'positive':
                    crossed = self.y[j - 1] >= target > self.y[j]
                else:
                    crossed = self.y[j - 1] <= target < self.y[j]
                if crossed:
                    t_tau = linear_interp_time(self.t[j-1], self.y[j-1], self.t[j], self.y[j], target)
                    y_tau = target
                    break

        # Decay segment: peak → event end only.
        # DO NOT extend past end_idx — that drags tau across the whole recording.
        # Add at most 3 extra samples as buffer for interpolation accuracy.
        decay_end = min(len(self.t), end_idx + 3)
        decay_result = fit_decay(self.t[gi:decay_end], self.y[gi:decay_end],
                                 baseline_val, amp, self.peak_direction)
        # Prefer fit_decay's 1/e crossing; fall back to detect_roi's crossing
        if decay_result.get('t_tau') is None:
            decay_result['t_tau'] = t_tau
            decay_result['y_tau'] = y_tau
        if decay_result.get('tau') is None and t_tau is not None:
            decay_result['tau'] = t_tau - tpk
            decay_result['model'] = '1/e interp'

        # AUC between t_on and t_end (above baseline only).
        # Interior samples are onset_idx+1 .. end_idx-1 (strictly inside the event).
        # Bookend with the exact interpolated crossing points at baseline_val.
        try:
            i0 = min(onset_idx + 1, len(self.t))
            i1 = max(end_idx, i0)
            interior_t = self.t[i0:i1]
            interior_y = self.y[i0:i1]
            seg_t = np.concatenate(([t_on], interior_t, [t_end]))
            seg_y = np.concatenate(([baseline_val], interior_y, [baseline_val]))
            auc = float(np.trapezoid(seg_y - baseline_val, seg_t)) if len(seg_t) > 1 else 0.0
        except Exception:
            auc = None

        peak_info = {
            't_peak': tpk, 'y_peak': ypk, 'idx_peak': gi,
            't_onset': t_on, 'y_onset': y_on, 't_end': t_end, 'y_end': y_end,
            't_tau': decay_result.get('t_tau'), 'y_tau': decay_result.get('y_tau'),
            'tau': decay_result.get('tau'), 'tau_fast': decay_result.get('tau_fast'),
            'tau_slow': decay_result.get('tau_slow'),
            'decay_model': decay_result.get('model'), 'decay_R2': decay_result.get('r2'),
            'amplitude': amp, 'baseline': baseline_val,
            'rise_time_10_90': rise_time, 'auc': auc
        }

        self.peaks.append(peak_info)
        self.info.insert(tk.END, f'Auto-detected 1 peak in ROI x[{x0:.4g},{x1:.4g}].\n')

    # ---------- filters ----------
    def apply_hp(self):
        if not SCIPY:
            messagebox.showerror('Missing scipy','Install scipy to use filters')
            return
        if self.t is None:
            return
        try:
            cutoff = float(self.hp_entry.get())
        except Exception:
            messagebox.showerror('Input','Enter numeric HP cutoff (Hz)')
            return
        fs = self._get_fs_hz()
        if not fs:
            messagebox.showerror('Fs','Sample rate unknown; load data or set Fs manually')
            return
        nyq = 0.5*fs
        if cutoff<=0 or cutoff>=nyq:
            messagebox.showerror('Cutoff','Cutoff must be between 0 and Nyquist')
            return
        b,a = butter(2, cutoff/nyq, btype='high')
        self.y = filtfilt(b,a,self.y)
        self.redraw()
        self.info.insert(tk.END, f'Applied HP {cutoff} Hz\n')

    def apply_lp(self):
        if not SCIPY:
            messagebox.showerror('Missing scipy','Install scipy to use filters')
            return
        if self.t is None:
            return
        try:
            cutoff = float(self.lp_entry.get())
        except Exception:
            messagebox.showerror('Input','Enter numeric LP cutoff (Hz)')
            return
        fs = self._get_fs_hz()
        if not fs:
            messagebox.showerror('Fs','Sample rate unknown; load data or set Fs manually')
            return
        nyq = 0.5*fs
        if cutoff<=0 or cutoff>=nyq:
            messagebox.showerror('Cutoff','Cutoff must be between 0 and Nyquist')
            return
        b,a = butter(2, cutoff/nyq, btype='low')
        self.y = filtfilt(b,a,self.y)
        self.redraw()
        self.info.insert(tk.END, f'Applied LP {cutoff} Hz\n')

    def reset_raw(self):
        if self.raw_y is None:
            return
        self.y = self.raw_y.copy()
        self.baseline_correction = None
        self.rolling_active = False
        self.rolling_var.set(False)
        self.redraw()
        self.info.insert(tk.END, 'Reset to raw data\n')

    # ---------- export / save ----------
    def export_events(self):
        if not self.peaks:
            messagebox.showwarning('No peaks','No peaks detected')
            return
        rows = []
        xu = self.x_unit
        auc_unit = f'{xu}·signal'
        for i, p in enumerate(self.peaks, start=1):
            rows.append({
                'event_#': i,
                f't_peak ({xu})': p['t_peak'],
                'y_peak': p['y_peak'],
                'amplitude': p['amplitude'],
                'baseline': p.get('baseline'),
                f't_onset ({xu})': p['t_onset'],
                f't_end ({xu})': p['t_end'],
                f't_tau ({xu})': p.get('t_tau'),
                f'tau ({xu})': p.get('tau'),
                f'tau_fast ({xu})': p.get('tau_fast'),
                f'tau_slow ({xu})': p.get('tau_slow'),
                'decay_model': p.get('decay_model'),
                'decay_R2': p.get('decay_R2'),
                f'rise_10_90 ({xu})': p['rise_time_10_90'],
                f'AUC ({auc_unit})': p['auc']
            })
        df = pd.DataFrame(rows)
        fname = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not fname:
            return
        df.to_csv(fname, index=False)
        self.info.insert(tk.END, f'Exported {len(rows)} events to {fname}\n')

    def show_summary(self):
        if not self.peaks:
            messagebox.showinfo('Summary Statistics', 'No peaks detected')
            return

        xu = self.x_unit

        def values(key):
            vals = [p.get(key) for p in self.peaks if p.get(key) is not None]
            return np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)

        def fmt(label, vals, unit=''):
            if vals.size == 0:
                return f'{label}: N/A'
            suffix = f' {unit}' if unit else ''
            return (
                f'{label}: mean={np.mean(vals):.6g}, median={np.median(vals):.6g}, '
                f'std={np.std(vals):.6g}{suffix} (n={len(vals)})'
            )

        msg = '\n'.join([
            f'Events detected: {len(self.peaks)}',
            '',
            fmt('Amplitude', values('amplitude')),
            fmt('Tau', values('tau'), xu),
            fmt('Rise 10-90', values('rise_time_10_90'), xu),
            fmt('AUC', values('auc'), f'{xu}·signal'),
        ])
        messagebox.showinfo('Summary Statistics', msg)

    def save_figure(self):
        fname = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
        if not fname:
            return
        self.fig.savefig(fname, dpi=200, facecolor=THEMES[self.theme_name]['fig_bg'])
        self.info.insert(tk.END, f'Figure saved to {fname}\n')

    # ---------- draw ----------
    def redraw(self):
        C = THEMES[self.theme_name]
        self.ax.clear()
        self.fig.patch.set_facecolor(C['fig_bg'])
        self.ax.set_facecolor(C['plot_bg'])
        self.ax.tick_params(colors=C['tick'])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(C['spine'])
        if self.t is None:
            self.ax.set_title('No data loaded', color=C['text'])
            self.canvas.draw()
            return
        self.ax.plot(self.t, self.y, lw=1, color=C['signal'], label='Signal')
        # baseline & threshold
        if self.h_baseline:
            self.ax.axhline(self.h_baseline.get_y(), color=C['baseline'], linestyle='--', linewidth=1, label='Baseline')
        if self.h_threshold:
            self.ax.axhline(self.h_threshold.get_y(), color=C['threshold'], linestyle='--', linewidth=1, label='Threshold')
        # ROIs
        for r in self.rois:
            x0,x1,y0,y1 = r
            self.ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False,
                                        linestyle=':', edgecolor=C['roi_edge'], linewidth=1.8))
        # peaks
        for p in self.peaks:
            self.ax.plot(p['t_peak'], p['y_peak'], marker='x', color='yellow', markersize=8)
            self.ax.plot(p['t_onset'], p['y_onset'], marker='x', color='green', markersize=8)
            if p.get('t_tau') is not None:
                self.ax.plot(p['t_tau'], p['y_tau'], marker='x', color='red', markersize=8)
        self.ax.set_xlabel(f'Time ({self.x_unit})', color=C['text'])
        self.ax.set_ylabel('Signal', color=C['text'])
        title = f'Trace - {len(self.peaks)} event(s)'
        if self.rolling_active:
            title += ' [rolling baseline corrected]'
        self.ax.set_title(title, color=C['text'])
        legend = self.ax.legend(loc='upper right', fontsize=8, facecolor=C['legend_face'],
                                edgecolor=C['legend_edge'])
        if legend:
            for text in legend.get_texts():
                text.set_color(C['legend_text'])
        self.canvas.draw()

    # ---------- params helpers ----------
    def apply_params(self):
        try:
            self.min_prominence = float(self.prom_entry.get())
        except Exception:
            self.min_prominence = 0.0
        try:
            self.min_distance = max(1, int(float(self.dist_entry.get())))
        except Exception:
            self.min_distance = 1
        self.info.insert(tk.END, f'Params applied: prominence={self.min_prominence}, min_distance={self.min_distance}\n')

    def undo_last_roi(self):
        """Remove the last ROI and any peaks which were detected within its x-range."""
        if not self.rois:
            self.info.insert(tk.END, 'No ROIs to undo.\n')
            return
        roi = self.rois.pop()          # remove last ROI
        x0, x1, y0, y1 = roi
        # remove peaks inside that ROI x-range (use peak time)
        before = len(self.peaks)
        self.peaks = [p for p in self.peaks if not (x0 <= p['t_peak'] <= x1)]
        removed = before - len(self.peaks)
        self.info.insert(tk.END, f'Undid last ROI x[{x0:.4g},{x1:.4g}]; removed {removed} peaks.\n')
        self.redraw()

    def delete_roi_by_index(self):
        """Ask user for ROI index and remove that ROI and peaks inside it."""
        if not self.rois:
            self.info.insert(tk.END, 'No ROIs to delete.\n')
            return
        # build a quick prompt message
        msg = 'Enter ROI index to remove (1..{}). Current ROIs:\\n'.format(len(self.rois))
        for i, r in enumerate(self.rois, start=1):
            msg += f'{i}: x[{r[0]:.4g},{r[1]:.4g}]\\n'
        try:
            s = simpledialog.askstring('Delete ROI', msg)
            if s is None:
                return
            idx = int(s) - 1
            if idx < 0 or idx >= len(self.rois):
                self.info.insert(tk.END, 'Invalid ROI index. Cancelled.\n')
                return
        except Exception:
            self.info.insert(tk.END, 'Invalid input. Cancelled.\n')
            return
        roi = self.rois.pop(idx)
        x0, x1, y0, y1 = roi
        before = len(self.peaks)
        self.peaks = [p for p in self.peaks if not (x0 <= p['t_peak'] <= x1)]
        removed = before - len(self.peaks)
        self.info.insert(tk.END, f'Removed ROI {idx+1} x[{x0:.4g},{x1:.4g}] and {removed} peaks.\n')
        self.redraw()

    def remove_peaks_in_last_roi(self):
        """Remove peaks that were detected inside the most recent ROI without removing the ROI itself."""
        if not self.rois:
            self.info.insert(tk.END, 'No ROIs exist.\n')
            return
        x0, x1, y0, y1 = self.rois[-1]
        before = len(self.peaks)
        self.peaks = [p for p in self.peaks if not (x0 <= p['t_peak'] <= x1)]
        removed = before - len(self.peaks)
        self.info.insert(tk.END, f'Removed {removed} peaks inside last ROI x[{x0:.4g},{x1:.4g}].\n')
        self.redraw()



def enable_dark_title_bar(root):
    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value)
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    enable_dark_title_bar(root)
    app = EventPickerApp(root)
    root.mainloop()
