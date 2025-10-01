#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# MDL Supererogatory Reasons Analyzer v2.4
# ============================================================================
# Implementazione fedele del criterio MDL per ragioni supererogatorie
# Riferimento: Coelati Rama, S. (2025). Teoria computazionale delle ragioni supererogatorie
#
# DIPENDENZE: Solo standard library Python (tkinter, csv, zlib, bz2, random)
# FUNZIONALITÀ:
# - Caricamento CSV con configurazione manuale tipo dati
# - Rilevamento automatico correlazioni negative + inversione
# - Soglie adattive basate su mediana (non più 0.5 fisso)
# - Overlap testi sintetici basato su correlazione diretta
# - Analisi MDL greedy per identificare ragioni necessarie vs supererogatorie
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import csv
import os
import zlib
import bz2
import random
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


# ============================================================================
# CORE MDL IMPLEMENTATION
# ============================================================================

def bytes_of(s: str) -> bytes:
    """Converte stringa in bytes per compressione. Usa 'replace' per gestire caratteri non-UTF8."""
    return s.encode('utf-8', errors='replace')


def compressed_len(data: bytes, method: str = 'zlib') -> int:
    """
    Calcola L(data) usando compressione come proxy per complessità di Kolmogorov K(·).
    zlib = veloce, buon compromesso (default)
    bz2 = compressione superiore ma più lento
    """
    if method == 'zlib':
        return len(zlib.compress(data, level=9))  # Level 9 = max compressione
    elif method == 'bz2':
        return len(bz2.compress(data, compresslevel=9))
    raise ValueError(f"Unknown compression method: {method}")


def L_cond(x: str, y: str, method: str = 'zlib') -> int:
    """
    Approssima K(x|y) usando compressione.
    Formula: L(x|y) ≈ L(y⊕x) - L(y)
    Separatore esplicito aiuta compressore a identificare confine tra y e x.
    """
    SEP = "\n§§§\n"  # Separatore inusuale per pattern compression
    L_y = compressed_len(bytes_of(y), method)
    L_yx = compressed_len(bytes_of(y + SEP + x), method)
    return max(0, L_yx - L_y)  # max(0, ...) evita negativi da fluttuazioni


def delta_hat(q: str, C: str, p: str, method: str = 'zlib') -> float:
    """
    Formula MDL centrale (equazione 2 tesi):
    Δ̂_C(p; q) = [L(q|C) - L(q|C,p)] / L(q|C)
    Misura riduzione complessità di q aggiungendo ragione p.
    Ritorna valore in [0,1]: 1 = massima riduzione, 0 = nessuna riduzione.
    """
    L_q_C = L_cond(q, C, method)
    if L_q_C == 0:
        return 0.0  # q già completamente determinato da C
    L_q_Cp = L_cond(q, C + "\n§§§\n" + p, method)
    reduction = (L_q_C - L_q_Cp) / L_q_C
    return max(-1.0, min(1.0, reduction))  # Clamp per fluttuazioni numeriche


def is_supererogatory(delta: float, tau: float = 0.15) -> bool:
    """
    Criterio supererogatorietà: p è supererogatoria quando Δ̂_C(p;q) < τ
    tau = 0.15 è valore empirico dalla tesi (margine rumore ~15%)
    """
    return delta < tau


# ============================================================================
# DATA TYPES
# ============================================================================

class ColumnType(Enum):
    """Enum type-safe per tipi di colonne CSV"""
    NUMERIC = "Numeric"
    BOOLEAN = "Boolean"
    CATEGORICAL = "Categorical"


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_correlation(values: List[float], target: List[float]) -> float:
    """
    Calcola correlazione di Pearson: r = Σ[(x-x̄)(y-ȳ)] / √[Σ(x-x̄)²·Σ(y-ȳ)²]
    Implementazione manuale per evitare dipendenze numpy/scipy.
    Ritorna valore in [-1, 1], o 0.0 se non calcolabile.
    """
    if len(values) != len(target) or len(values) < 2:
        return 0.0
    
    n = len(values)
    mean_x = sum(values) / n
    mean_y = sum(target) / n
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, target))
    denom_x = sum((x - mean_x) ** 2 for x in values) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in target) ** 0.5
    
    if denom_x == 0 or denom_y == 0:
        return 0.0  # Varianza nulla
    
    return numerator / (denom_x * denom_y)


def compute_median(values: List[float]) -> float:
    """
    Calcola mediana. Più robusta agli outlier rispetto a media.
    Fornisce soglia naturale per dividere high/low senza bias.
    """
    if not values:
        return 0.5  # Fallback per dati normalizzati
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2.0
    else:
        return sorted_vals[n//2]


def analyze_columns(rows: List[Dict], target_col: str, threshold: Any,
                   col_type: ColumnType, reason_cols: List[str]) -> Dict[str, Dict]:
    """
    Analizza colonne ragione: correlazione + mediana + flag inversione.
    Pipeline: 1) Costruisci vettore successo, 2) Per ogni colonna calcola stats
    Ritorna: Dict[col_name, {'correlation': float, 'is_inverted': bool, 'median': float}]
    Inversione automatica se corr < -0.1 (correlazione negativa)
    """
    # Costruisci vettore successo (1.0 = pass, 0.0 = fail)
    success_values = []
    for row in rows:
        try:
            val_str = str(row.get(target_col, '')).strip()
            
            if col_type == ColumnType.NUMERIC:
                is_success = float(val_str) >= float(threshold)
            elif col_type == ColumnType.BOOLEAN:
                val_norm = val_str.lower()
                threshold_norm = str(threshold).lower()
                true_vals = {'yes', 'y', 'true', 't', '1', 'si', 'sì'}
                val_bool = val_norm in true_vals
                thresh_bool = threshold_norm in true_vals
                is_success = val_bool == thresh_bool
            else:  # CATEGORICAL
                is_success = val_str == str(threshold)
            
            success_values.append(1.0 if is_success else 0.0)
        except:
            success_values.append(0.0)
    
    # Analizza ogni colonna ragione
    analysis = {}
    for col in reason_cols:
        try:
            # Estrai valori numerici
            col_values = []
            valid_indices = []
            for i, row in enumerate(rows):
                try:
                    val = float(row.get(col, 0))
                    col_values.append(val)
                    valid_indices.append(i)
                except:
                    pass
            
            if len(col_values) < 10:  # Troppo pochi dati numerici
                analysis[col] = {'correlation': 0.0, 'is_inverted': False, 'median': 0.5}
                continue
            
            filtered_success = [success_values[i] for i in valid_indices]
            
            # Calcola stats
            corr = compute_correlation(col_values, filtered_success)
            median = compute_median(col_values)
            is_inverted = (corr < -0.1)  # Soglia conservativa per correlazioni negative
            
            analysis[col] = {
                'correlation': abs(corr),  # Usa valore assoluto per overlap
                'is_inverted': is_inverted,
                'median': median
            }
            
        except Exception:
            analysis[col] = {'correlation': 0.0, 'is_inverted': False, 'median': 0.5}
    
    return analysis


# ============================================================================
# SYNTHETIC TEXT GENERATION
# ============================================================================

def make_text_representation(core_tokens: List[str], other_tokens: List[str],
                            core_share: float, total_len: int, rng: random.Random) -> str:
    """
    Genera testo sintetico con overlap semantico controllato.
    core_tokens = vocabolario "successo", other_tokens = vocabolario neutro
    core_share = frazione [0,1] di token core nel testo finale
    Shuffle rompe pattern sequenziali per compressione pulita.
    """
    total_len = max(100, total_len)  # Minimo 100 token per robustezza
    n_core = max(0, min(total_len, int(round(total_len * core_share))))
    n_other = total_len - n_core
    
    tokens = [rng.choice(core_tokens) for _ in range(n_core)] + \
             [rng.choice(other_tokens) for _ in range(n_other)]
    rng.shuffle(tokens)
    return " ".join(tokens)


def build_mdl_texts_from_data(rows: List[Dict], target_col: str, threshold: Any,
                              col_type: ColumnType, reason_cols: List[str], 
                              analysis: Dict[str, Dict], seed: int = 42) -> Dict:
    """
    Costruisce rappresentazioni testuali MDL da CSV.
    Architettura: q (conclusione), B (background), E (evidence), p_i (ragioni), r_dom (test)
    INNOVAZIONE v2.4: Overlap basato direttamente su |correlazione| invece di success_rate
    Formula: overlap = 0.30 + 0.60 * |correlation|
    Seed fisso garantisce riproducibilità.
    """
    rng = random.Random(seed)
    
    # Vocabolario successo (parole associate a performance positiva)
    SUCCESS_VOCAB = [
        "achieve", "succeed", "attain", "accomplish", "excel", "master", "proficient",
        "capable", "competent", "effective", "successful", "strong", "solid", "robust",
        "reliable", "consistent", "outstanding", "high-quality", "optimal", "superior"
    ]
    
    # Vocabolario neutro esteso
    ALL_VOCAB = SUCCESS_VOCAB + [f"neutral{i}" for i in range(300)]
    CONTRAST_VOCAB = [t for t in ALL_VOCAB if t not in SUCCESS_VOCAB]
    
    # Funzione helper per determinare successo
    def is_success(row: Dict) -> bool:
        try:
            val_str = str(row.get(target_col, '')).strip()
            if not val_str:
                return False
            
            if col_type == ColumnType.NUMERIC:
                try:
                    return float(val_str) >= float(threshold)
                except:
                    return False
            elif col_type == ColumnType.BOOLEAN:
                val_norm = val_str.lower()
                threshold_norm = str(threshold).lower()
                true_vals = {'yes', 'y', 'true', 't', '1', 'si', 'sì'}
                val_bool = val_norm in true_vals
                thresh_bool = threshold_norm in true_vals
                return val_bool == thresh_bool
            else:  # CATEGORICAL
                return val_str == str(threshold)
        except Exception:
            return False
    
    # Genera testi con overlap controllato
    q_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, 0.92, 1000, rng)  # Alto overlap
    B_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, 0.18, 450, rng)   # Basso overlap
    E_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, 0.14, 350, rng)   # Overlap minimo
    
    # Genera rappresentazioni ragioni
    reasons = []
    labels = []
    for col in reason_cols:
        try:
            col_analysis = analysis.get(col, {'correlation': 0.0, 'is_inverted': False, 'median': 0.5})
            correlation = col_analysis['correlation']  # Già abs()
            is_inverted = col_analysis['is_inverted']
            
            # Overlap diretto da correlazione (INNOVAZIONE v2.4)
            overlap = max(0.15, min(0.94, 0.30 + 0.60 * correlation))
            
            p_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, overlap, 380, rng)
            reasons.append(p_text)
            
            # Label con prefix LOW se invertita
            label_prefix = f"LOW {col}" if is_inverted else f"{col}"
            labels.append(f"{label_prefix} (corr={correlation:.2f}, ovl≈{overlap:.2f})")
        except Exception:
            p_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, 0.3, 380, rng)
            reasons.append(p_text)
            labels.append(f"{col} (overlap≈0.30)")
    
    # Ragione dominante per test non-monotonicità
    r_dom_text = make_text_representation(SUCCESS_VOCAB, CONTRAST_VOCAB, 0.88, 400, rng)
    
    return {
        'q': q_text,
        'B': B_text,
        'E': E_text,
        'reasons': reasons,
        'labels': labels,
        'r_dom': r_dom_text
    }


def _reason_satisfied(row: Dict, col: str, is_inverted: bool = False, median_val: float = 0.5) -> bool:
    """
    Determina se ragione soddisfatta per una riga.
    INNOVAZIONE v2.4: Usa MEDIANA invece di 0.5 fisso
    Esempio: study_hours >= 3.5h (mediana) invece di >= 0.5h (fisso errato)
    Inversione per correlazioni negative (es. LOW social_media = positivo)
    """
    try:
        val = str(row.get(col, '')).strip().lower()
        
        # Controlla valori booleani espliciti
        if val in ('yes', 'y', 'true', '1', 'high', 'good', 'excellent', 'si', 'sì', 't'):
            result = True
        elif val in ('no', 'n', 'false', '0', 'low', 'poor', 'bad', 'f'):
            result = False
        else:
            # Confronto numerico con MEDIANA (non 0.5 fisso)
            try:
                numeric_val = float(val)
                result = numeric_val >= median_val  # CRUCIALE: usa mediana specifica colonna
            except:
                result = False
        
        # Inverti se correlazione negativa
        if is_inverted:
            result = not result
        
        return result
    except Exception:
        return False


# ============================================================================
# GREEDY MDL SELECTION ALGORITHM
# ============================================================================

def greedy_mdl_selection(q: str, C_base: str, reasons: List[str],
                        tau: float, method: str = 'zlib') -> Tuple[List[int], List[float], List[float]]:
    """
    Algoritmo greedy per selezione set minimo S* di ragioni necessarie.
    Loop: seleziona ragione con max Δ̂, aggiungi a S*, aggiorna contesto, ripeti fino a max Δ̂ < τ
    Tie-breaking deterministico: preferisce indice minore se Δ̂ uguali
    Ritorna: (indici selezionati, Δ̂ al momento selezione, Δ̂ post-selezione)
    """
    EPS = 1e-12  # Tolleranza numerica per confronti float
    n = len(reasons)
    remaining = list(range(n))
    selected = []
    marginal_gains = [0.0] * n
    C = C_base
    
    # Loop principale selezione greedy
    while remaining:
        best_i = None
        best_delta = 0.0
        
        # Trova ragione con max Δ̂
        for i in remaining:
            delta = delta_hat(q, C, reasons[i], method)
            if delta > best_delta + EPS or (abs(delta - best_delta) <= EPS and (best_i is None or i < best_i)):
                best_delta = delta
                best_i = i
        
        # Stop se max Δ̂ < τ
        if best_i is None or best_delta < tau - EPS:
            break
        
        # Aggiungi a S* e aggiorna contesto
        selected.append(best_i)
        marginal_gains[best_i] = best_delta
        C = C + "\n§§§\n" + reasons[best_i]
        remaining.remove(best_i)
    
    # Calcola Δ̂ post-selezione per tutte le ragioni
    C_final = C_base
    for i in selected:
        C_final = C_final + "\n§§§\n" + reasons[i]
    post_marginal = [delta_hat(q, C_final, reasons[i], method) for i in range(n)]
    
    return selected, marginal_gains, post_marginal


# ============================================================================
# ADAPTIVE TAU (ELBOW METHOD)
# ============================================================================

def adaptive_tau(deltas: List[float], min_tau: float = 0.02, max_tau: float = 0.40) -> float:
    """
    Stima τ adattivo da profilo Δ̂ usando metodo del gomito.
    Trova punto di massimo cambio pendenza (gap) nella curva ordinata di Δ̂.
    Clamp in [0.02, 0.40], fallback 0.15 se non determinabile.
    """
    if len(deltas) < 2:
        return 0.15
    
    sorted_d = sorted([max(0.0, min(1.0, d)) for d in deltas], reverse=True)
    gaps = [sorted_d[i] - sorted_d[i+1] for i in range(len(sorted_d)-1)]
    
    if not gaps:
        return 0.15
    
    i_star = max(range(len(gaps)), key=lambda i: gaps[i])
    tau_est = (sorted_d[i_star] + sorted_d[i_star+1]) / 2.0
    
    return max(min_tau, min(max_tau, tau_est))


# ============================================================================
# GUI APPLICATION
# ============================================================================

class MDLApp(tk.Tk):
    """
    Applicazione GUI principale.
    Gestisce caricamento CSV, configurazione, analisi MDL, visualizzazione risultati.
    """
    
    # Palette colori professionale
    COLORS = {
        'bg': '#f5f5f5',
        'fg': '#2c3e50',
        'accent': '#3498db',
        'success': '#27ae60',
        'warning': '#e67e22',
        'danger': '#e74c3c',
        'border': '#bdc3c7',
        'highlight': '#ecf0f1',
        'necessary': '#2ecc71',
        'supererogatory': '#f39c12'
    }
    
    def __init__(self):
        """Inizializza app, setup UI, avvia caricamento CSV"""
        super().__init__()
        
        # Configurazione finestra
        self.title("MDL Supererogatory Reasons Analyzer v2.4")
        self.geometry("1320x920")
        self.minsize(1100, 760)
        self.configure(bg=self.COLORS['bg'])
        
        # Variabili di stato
        self.csv_path: Optional[str] = None
        self.rows: List[Dict] = []
        self.target_col: Optional[str] = None
        self.threshold: Any = None
        self.col_type: Optional[ColumnType] = None
        self.reason_cols: List[str] = []
        self.analysis: Dict[str, Dict] = {}
        self.mdl_data: Optional[Dict] = None
        self.deltas_base: List[float] = []
        self.selected: List[int] = []
        self.marginal_gains: List[float] = []
        self.post_marginal: List[float] = []
        self.tau_value: float = 0.15
        self.method = tk.StringVar(value='zlib')
        self.include_r = tk.BooleanVar(value=False)
        self.sort_bars = tk.BooleanVar(value=True)
        self.mdl_executed = False
        
        # Setup
        self._setup_styles()
        self._build_ui()
        self.after(100, self._load_csv_dialog)
    
    def _setup_styles(self):
        """Configura stili ttk per look moderno"""
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        self.style.configure('TFrame', background=self.COLORS['bg'])
        self.style.configure('TLabel', background=self.COLORS['bg'], foreground=self.COLORS['fg'])
        self.style.configure('TButton', padding=8, relief='flat', background=self.COLORS['accent'])
        self.style.map('TButton', background=[('active', self.COLORS['success'])])
        self.style.configure('Accent.TButton', background=self.COLORS['success'])
        self.style.configure('TCheckbutton', background=self.COLORS['bg'], foreground=self.COLORS['fg'])
        self.style.configure('TNotebook', background=self.COLORS['bg'], borderwidth=0)
        self.style.configure('TNotebook.Tab', padding=[12, 6])
        
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        
        heading_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.heading_font = heading_font
    
    def _build_ui(self):
        """Costruisce interfaccia completa: header + notebook tabs + status bar"""
        main_container = ttk.Frame(self, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = self._build_header(main_container)
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Notebook tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Analysis
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="📊 Analysis")
        self._build_analysis_tab()
        
        # Tab 2: Data
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="📋 Data")
        self._build_data_tab()
        
        # Tab 3: Theory
        self.tab_theory = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_theory, text="📖 Theory")
        self._build_theory_tab()
        
        # Status bar
        self.status_bar = ttk.Label(main_container, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def _build_header(self, parent):
        """Header: titolo + pulsante Load CSV"""
        header = ttk.Frame(parent)
        
        title_label = ttk.Label(header, text="MDL Supererogatory Reasons Analyzer",
                               font=self.heading_font, foreground=self.COLORS['accent'])
        title_label.pack(side=tk.LEFT)
        
        btn_load = ttk.Button(header, text="📁 Load CSV", command=self._load_csv_dialog)
        btn_load.pack(side=tk.RIGHT, padx=2)
        
        return header
    
    def _build_analysis_tab(self):
        """Tab Analysis: controlli + chart + panel risultati"""
        # Control panel
        ctrl_frame = ttk.LabelFrame(self.tab_analysis, text="Controls", padding=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Row 1: opzioni
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Compression:").pack(side=tk.LEFT)
        method_cb = ttk.Combobox(row1, textvariable=self.method, values=['zlib', 'bz2'],
                                width=8, state='readonly')
        method_cb.pack(side=tk.LEFT, padx=(4, 12))
        method_cb.bind('<<ComboboxSelected>>', lambda e: self._invalidate_results())
        
        ttk.Checkbutton(row1, text="Include dominant reason r (non-monotonicity test)",
                       variable=self.include_r, command=self._invalidate_results).pack(side=tk.LEFT, padx=8)
        
        ttk.Checkbutton(row1, text="Sort by Δ̂", variable=self.sort_bars,
                       command=self._redraw_chart).pack(side=tk.LEFT, padx=8)
        
        # Row 2: Execute button
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=6)
        
        self.btn_execute = ttk.Button(row2, text="▶ Execute MDL Analysis",
                                      command=self._execute_mdl, style='Accent.TButton')
        self.btn_execute.pack(side=tk.LEFT)
        self.btn_execute.config(state='disabled')
        
        # Content: chart + panel
        content = ttk.Frame(self.tab_analysis)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Chart canvas
        chart_frame = ttk.LabelFrame(content, text="Δ̂ Profile", padding=5)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(chart_frame, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self._redraw_chart())
        
        # Results panel
        results_frame = ttk.Frame(content, width=400)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y)
        results_frame.pack_propagate(False)
        
        # Context
        self.lbl_context = ttk.Label(results_frame, text="Context: C = ⟨B, E⟩",
                                    font=('Segoe UI', 10, 'italic'))
        self.lbl_context.pack(anchor=tk.W, pady=(0, 8))
        
        # Tau
        tau_frame = ttk.LabelFrame(results_frame, text="Threshold", padding=8)
        tau_frame.pack(fill=tk.X, pady=4)
        self.lbl_tau = ttk.Label(tau_frame, text="τ = 0.150 (default)", font=('Segoe UI', 10))
        self.lbl_tau.pack(anchor=tk.W)
        
        # Necessary reasons
        necessary_frame = ttk.LabelFrame(results_frame, text="✓ Necessary Reasons (S*)", padding=8)
        necessary_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.txt_necessary = tk.Text(necessary_frame, height=8, wrap=tk.WORD, font=('Consolas', 9),
                                    bg=self.COLORS['highlight'], relief=tk.FLAT)
        self.txt_necessary.pack(fill=tk.BOTH, expand=True)
        self.txt_necessary.insert('1.0', "Press 'Execute MDL Analysis' to start")
        self.txt_necessary.config(state=tk.DISABLED)
        
        # Supererogatory reasons
        super_frame = ttk.LabelFrame(results_frame, text="⊘ Supererogatory Reasons", padding=8)
        super_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.txt_supererogatory = tk.Text(super_frame, height=8, wrap=tk.WORD, font=('Consolas', 9),
                                          bg=self.COLORS['highlight'], relief=tk.FLAT)
        self.txt_supererogatory.pack(fill=tk.BOTH, expand=True)
        self.txt_supererogatory.config(state=tk.DISABLED)
        
        # Legend
        legend_frame = ttk.LabelFrame(results_frame, text="Legend", padding=8)
        legend_frame.pack(fill=tk.X, pady=4)
        
        legend_text = (
            "• Green bars with ①: Necessary reasons (Δ̂ ≥ τ)\n"
            "• Orange bars: Supererogatory (Δ̂ < τ)\n"
            "• Red dashed line: τ threshold\n"
            "• Label: name (corr=correlation, ovl=overlap)\n"
            "• LOW prefix: inverted (negative correlation)"
        )
        ttk.Label(legend_frame, text=legend_text, justify=tk.LEFT, font=('Segoe UI', 9)).pack(anchor=tk.W)
    
    def _build_data_tab(self):
        """Tab Data: preview CSV in tabella"""
        info_frame = ttk.Frame(self.tab_data, padding=10)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_data_info = ttk.Label(info_frame, text="No data loaded", font=self.heading_font)
        self.lbl_data_info.pack(anchor=tk.W)
        
        tree_frame = ttk.Frame(self.tab_data, padding=10)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, selectmode='browse')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(self.tab_data, orient=tk.HORIZONTAL, command=self.tree.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=hsb.set)
    
    def _build_theory_tab(self):
        """Tab Theory: spiegazione teorica MDL"""
        text_frame = ttk.Frame(self.tab_theory, padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        theory_text = tk.Text(text_frame, wrap=tk.WORD, font=('Georgia', 10), padx=10, pady=10)
        theory_text.pack(fill=tk.BOTH, expand=True)
        
        content = """MDL CRITERION FOR SUPEREROGATORY REASONS
========================================

Theory (Coelati Rama, 2025)
----------------------------
A reason p is SUPEREROGATORY for conclusion q in context C when:

    K(q|C,p) / K(q|C) > 1 - τ

Equivalently: Δ̂_C(p; q) = [L(q|C) - L(q|C,p)] / L(q|C) < τ

Context Structure: C = ⟨B, R, E⟩
- B: Background beliefs
- R: Already accepted reasons
- E: Environmental evidence

Algorithm
---------
1. Compute Δ̂_C(p_i; q) for all reasons
2. Estimate adaptive τ (elbow method)
3. Greedy selection: add reason with max Δ̂ to S*, update C, repeat until max Δ̂ < τ
4. Classify: S* = necessary, others = supererogatory

v2.4 Improvements
-----------------
- Adaptive median thresholds (not fixed 0.5)
- Correlation-based text overlap
- Automatic negative correlation inversion

References
----------
Coelati Rama, S. (2025). Teoria computazionale delle ragioni supererogatorie.
Crupi, V. & Iacona, A. (2023). Outline of a theory of reasons.
Grünwald, P. (2007). The Minimum Description Length Principle.
"""
        
        theory_text.insert('1.0', content.strip())
        theory_text.config(state=tk.DISABLED)
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def _load_csv_dialog(self):
        """Apre file dialog per selezione CSV"""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not path:
            if not self.csv_path:
                messagebox.showwarning("No Data", "No CSV loaded. Please load a file to proceed.")
            return
        
        try:
            self._load_csv(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load CSV:\n{str(e)}\n\nPlease check the file format.")
            import traceback
            traceback.print_exc()
            return
    
    def _load_csv(self, path: str):
        """Carica CSV, configura, analizza, genera testi MDL"""
        # Leggi CSV
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        
        if not self.rows:
            raise ValueError("CSV file is empty")
        
        self.csv_path = path
        columns = list(self.rows[0].keys())
        
        # ConfigDialog per setup
        config_dlg = ConfigDialog(self, columns, self.rows)
        if not config_dlg.result:
            return
        
        self.target_col = config_dlg.result['target']
        self.threshold = config_dlg.result['threshold']
        self.col_type = config_dlg.result['col_type']
        self.reason_cols = config_dlg.result['reasons']
        
        # Analizza colonne
        self._status("Analyzing columns (correlation + median)...")
        self.update_idletasks()
        
        self.analysis = analyze_columns(
            self.rows, self.target_col, self.threshold, self.col_type, self.reason_cols
        )
        
        # Mostra analisi
        analysis_lines = []
        for col in self.reason_cols:
            col_info = self.analysis.get(col, {})
            corr = col_info.get('correlation', 0.0)
            inv = col_info.get('is_inverted', False)
            med = col_info.get('median', 0.5)
            prefix = "LOW " if inv else ""
            analysis_lines.append(f"{prefix}{col}: corr={corr:.3f}, median={med:.2f}")
        
        analysis_msg = "Column Analysis:\n\n" + "\n".join(analysis_lines)
        messagebox.showinfo("Correlation & Median Analysis", analysis_msg)
        
        # Genera testi MDL
        try:
            self.mdl_data = build_mdl_texts_from_data(
                self.rows, self.target_col, self.threshold, self.col_type, 
                self.reason_cols, self.analysis
            )
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process data:\n{str(e)}")
            return
        
        # Update GUI
        self._populate_data_table()
        self._update_data_info()
        self.btn_execute.config(state='normal')
        self._invalidate_results()
        self._status(f"Loaded {len(self.rows)} rows from {os.path.basename(path)}")
    
    def _populate_data_table(self):
        """Popola treeview con primi 100 righe CSV"""
        try:
            self.tree.delete(*self.tree.get_children())
            
            all_cols = [self.target_col] + self.reason_cols
            self.tree['columns'] = all_cols
            self.tree['show'] = 'headings'
            
            for col in all_cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor=tk.CENTER)
            
            for row in self.rows[:100]:
                values = [row.get(col, '') for col in all_cols]
                self.tree.insert('', tk.END, values=values)
        except Exception as e:
            print(f"Error populating table: {e}")
    
    def _update_data_info(self):
        """Aggiorna label info dataset"""
        try:
            threshold_str = str(self.threshold)
            if self.col_type == ColumnType.NUMERIC:
                threshold_str = f"≥ {self.threshold}"
            elif self.col_type == ColumnType.BOOLEAN:
                threshold_str = f"= {self.threshold}"
            else:
                threshold_str = f"= '{self.threshold}'"
            
            n_inv = sum(1 for col_info in self.analysis.values() if col_info.get('is_inverted', False))
            inv_str = f" | {n_inv} inverted" if n_inv > 0 else ""
            
            info = f"Dataset: {os.path.basename(self.csv_path)} | {len(self.rows)} rows | Target: {self.target_col} {threshold_str} ({self.col_type.value}){inv_str}"
            self.lbl_data_info.config(text=info)
        except Exception as e:
            self.lbl_data_info.config(text=f"Dataset: {os.path.basename(self.csv_path)} | {len(self.rows)} rows")
    
    # ========================================================================
    # MDL EXECUTION
    # ========================================================================
    
    def _execute_mdl(self):
        """Esegue pipeline MDL completa"""
        if not self.mdl_data:
            messagebox.showerror("No Data", "Please load CSV first")
            return
        
        self._status("Executing MDL analysis...")
        self.btn_execute.config(state='disabled')
        self.update_idletasks()
        
        try:
            # Costruisci contesto
            C = self.mdl_data['B'] + "\n§§§\n" + self.mdl_data['E']
            if self.include_r.get():
                C = C + "\n§§§\n" + self.mdl_data['r_dom']
            
            q = self.mdl_data['q']
            reasons = self.mdl_data['reasons']
            method = self.method.get()
            
            # Calcola delta base
            self.deltas_base = [delta_hat(q, C, p, method) for p in reasons]
            
            # Tau adattivo
            self.tau_value = adaptive_tau(self.deltas_base)
            
            # Greedy selection
            self.selected, self.marginal_gains, self.post_marginal = greedy_mdl_selection(
                q, C, reasons, self.tau_value, method
            )
            
            self.mdl_executed = True
            
            # Update UI
            self._update_context_label()
            self._update_tau_label()
            self._update_results()
            self._redraw_chart()
            
            n_nec = len(self.selected)
            n_sup = len(reasons) - n_nec
            self._status(f"Analysis complete: {n_nec} necessary, {n_sup} supererogatory | τ = {self.tau_value:.3f}")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"MDL execution failed:\n{str(e)}")
            self._status("Error")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_execute.config(state='normal')
    
    def _invalidate_results(self):
        """Invalida risultati dopo cambio parametri"""
        self.mdl_executed = False
        self.selected = []
        self.marginal_gains = []
        self.post_marginal = []
        
        self._update_context_label()
        self.lbl_tau.config(text="τ = 0.150 (default)")
        
        self.txt_necessary.config(state=tk.NORMAL)
        self.txt_necessary.delete('1.0', tk.END)
        self.txt_necessary.insert('1.0', "Press 'Execute MDL Analysis' to compute")
        self.txt_necessary.config(state=tk.DISABLED)
        
        self.txt_supererogatory.config(state=tk.NORMAL)
        self.txt_supererogatory.delete('1.0', tk.END)
        self.txt_supererogatory.config(state=tk.DISABLED)
        
        self._redraw_chart()
    
    def _update_context_label(self):
        """Aggiorna label contesto"""
        if self.include_r.get():
            self.lbl_context.config(text="Context: C = ⟨B, R, E⟩ with dominant r")
        else:
            self.lbl_context.config(text="Context: C = ⟨B, E⟩")
    
    def _update_tau_label(self):
        """Aggiorna label tau"""
        self.lbl_tau.config(text=f"τ = {self.tau_value:.3f} (adaptive)")
    
    def _update_results(self):
        """Aggiorna text areas con risultati"""
        try:
            labels = self.mdl_data['labels']
            
            # Necessary
            self.txt_necessary.config(state=tk.NORMAL)
            self.txt_necessary.delete('1.0', tk.END)
            if self.selected:
                for rank, i in enumerate(self.selected, 1):
                    name = labels[i].split(' (')[0]
                    delta = self.marginal_gains[i]
                    self.txt_necessary.insert(tk.END, f"{rank}. {name}\n   Δ̂ = {delta:.3f} ({delta*100:.1f}%)\n\n")
            else:
                self.txt_necessary.insert(tk.END, "None (all reasons below τ)")
            self.txt_necessary.config(state=tk.DISABLED)
            
            # Supererogatory
            self.txt_supererogatory.config(state=tk.NORMAL)
            self.txt_supererogatory.delete('1.0', tk.END)
            super_indices = [i for i in range(len(labels)) if i not in self.selected]
            if super_indices:
                for i in super_indices:
                    name = labels[i].split(' (')[0]
                    delta_base = self.deltas_base[i]
                    delta_post = self.post_marginal[i]
                    self.txt_supererogatory.insert(tk.END, 
                        f"• {name}\n  Δ̂_base = {delta_base:.3f}, Δ̂_post = {delta_post:.3f}\n\n")
            else:
                self.txt_supererogatory.insert(tk.END, "None (all reasons necessary)")
            self.txt_supererogatory.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error updating results: {e}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def _redraw_chart(self):
        """Disegna chart barre con Δ̂ profile"""
        try:
            self.canvas.delete('all')
            
            if not self.deltas_base:
                self.canvas.create_text(400, 300, text="No data to display",
                                       font=('Segoe UI', 14), fill='gray')
                return
            
            W = self.canvas.winfo_width() or 800
            H = self.canvas.winfo_height() or 600
            
            PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 60, 100
            plot_w = W - PAD_L - PAD_R
            plot_h = H - PAD_T - PAD_B
            
            # Ordina dati
            n = len(self.deltas_base)
            indices = list(range(n))
            if self.sort_bars.get():
                indices.sort(key=lambda i: self.deltas_base[i], reverse=True)
            
            labels = [self.mdl_data['labels'][i].split(' (')[0] for i in indices]
            deltas = [max(0, min(1, self.deltas_base[i])) for i in indices]
            
            # Assi
            self.canvas.create_line(PAD_L, H-PAD_B, W-PAD_R, H-PAD_B, fill='black', width=2)
            self.canvas.create_line(PAD_L, H-PAD_B, PAD_L, PAD_T, fill='black', width=2)
            
            # Griglia Y
            for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                y = H - PAD_B - y_val * plot_h
                self.canvas.create_line(PAD_L, y, W-PAD_R, y, fill='#e0e0e0', dash=(2,4))
                self.canvas.create_text(PAD_L-10, y, text=f"{y_val:.2f}", anchor=tk.E, font=('Segoe UI', 9))
            
            # Linea tau
            if self.mdl_executed:
                y_tau = H - PAD_B - self.tau_value * plot_h
                self.canvas.create_line(PAD_L, y_tau, W-PAD_R, y_tau,
                                       fill=self.COLORS['danger'], width=2, dash=(6,4))
                self.canvas.create_text(W-PAD_R-10, y_tau-8, text=f"τ = {self.tau_value:.2f}",
                                       anchor=tk.E, fill=self.COLORS['danger'], font=('Segoe UI', 9, 'bold'))
            
            # Barre
            bar_w = max(20, plot_w / (2*n))
            for rank, (i, delta) in enumerate(zip(indices, deltas), 1):
                x_center = PAD_L + (rank - 0.5) * (plot_w / n)
                x0 = x_center - bar_w/2
                x1 = x_center + bar_w/2
                y1 = H - PAD_B
                y0 = y1 - delta * plot_h
                
                is_nec = self.mdl_executed and i in self.selected
                color = self.COLORS['necessary'] if is_nec else self.COLORS['supererogatory']
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='white', width=1)
                
                # Badge
                if is_nec:
                    rank_in_s = self.selected.index(i) + 1
                    badge_y = y0 - 16
                    self.canvas.create_oval(x0+4, badge_y-10, x0+24, badge_y+10,
                                           fill=self.COLORS['necessary'], outline='white', width=2)
                    self.canvas.create_text(x0+14, badge_y, text=str(rank_in_s),
                                           fill='white', font=('Segoe UI', 10, 'bold'))
                
                # Value labels
                label_y = y0 - (28 if is_nec else 10)
                if self.mdl_executed and len(self.post_marginal) > i:
                    delta_post = self.post_marginal[i] if i not in self.selected else self.marginal_gains[i]
                    label_text = f"{delta*100:.0f} | {delta_post*100:.0f}"
                else:
                    label_text = f"{delta*100:.0f}%"
                
                self.canvas.create_text(x_center, label_y, text=label_text,
                                       font=('Segoe UI', 9, 'bold'))
                
                # X labels
                label_lines = self._wrap_text(labels[rank-1], 15)
                label_y_start = H - PAD_B + 15
                for line_i, line in enumerate(label_lines):
                    self.canvas.create_text(x_center, label_y_start + line_i*12,
                                           text=line, font=('Segoe UI', 8))
            
            # Axis labels
            self.canvas.create_text(PAD_L//2, (PAD_T + H-PAD_B)//2, text="Δ̂_C(p; q)",
                                   angle=90, font=('Segoe UI', 11, 'bold'))
            self.canvas.create_text((PAD_L + W-PAD_R)//2, H-30, text="Reasons",
                                   font=('Segoe UI', 11, 'bold'))
            self.canvas.create_text((PAD_L + W-PAD_R)//2, 30,
                                   text="Reduction Profile (base | post-S*)",
                                   font=('Segoe UI', 12, 'bold'))
        except Exception as e:
            print(f"Error drawing chart: {e}")
    
    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Word wrap per labels"""
        words = text.split()
        lines = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > width:
                if current:
                    lines.append(' '.join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            lines.append(' '.join(current))
        return lines or [""]
    
    def _status(self, msg: str):
        """Aggiorna status bar"""
        self.status_bar.config(text=msg)


# ============================================================================
# CONFIG DIALOG
# ============================================================================

class ConfigDialog(tk.Toplevel):
    """Dialog configurazione: target, tipo, threshold, ragioni"""
    
    def __init__(self, parent, columns: List[str], rows: List[Dict]):
        super().__init__(parent)
        self.result = None
        self.columns = columns
        self.rows = rows
        
        self.title("Configure Analysis")
        self.geometry("650x600")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.target_var = tk.StringVar()
        self.type_var = tk.StringVar(value='Numeric')
        self.type_var.trace('w', self._on_type_change)
        self.threshold_var = tk.StringVar(value='50')
        self.reason_vars = {}
        
        self.threshold_widget_container = None
        
        self._build_ui()
        
        # Center
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        self.wait_window()
    
    def _build_ui(self):
        """Costruisce UI dialog"""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Configure MDL Analysis", font=('Segoe UI', 14, 'bold')).pack(anchor=tk.W, pady=(0,15))
        
        # Target column
        target_frame = ttk.LabelFrame(main_frame, text="1. Select Target Column (q)", padding=10)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="This column represents the conclusion to explain:").pack(anchor=tk.W)
        target_cb = ttk.Combobox(target_frame, textvariable=self.target_var, values=self.columns, state='readonly', width=50)
        target_cb.pack(fill=tk.X, pady=(5,0))
        if self.columns:
            target_cb.current(0)
        
        # Type + Threshold
        type_thresh_frame = ttk.LabelFrame(main_frame, text="2. Data Type & Success Threshold", padding=10)
        type_thresh_frame.pack(fill=tk.X, pady=5)
        
        row_type = ttk.Frame(type_thresh_frame)
        row_type.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(row_type, text="Data Type:").pack(side=tk.LEFT)
        type_cb = ttk.Combobox(row_type, textvariable=self.type_var, 
                              values=[t.value for t in ColumnType], 
                              state='readonly', width=15)
        type_cb.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(type_thresh_frame, text="Success Threshold:").pack(anchor=tk.W, pady=(0, 5))
        self.threshold_widget_container = ttk.Frame(type_thresh_frame)
        self.threshold_widget_container.pack(fill=tk.X)
        
        self._on_type_change()
        
        # Reason columns
        reason_frame = ttk.LabelFrame(main_frame, text="3. Select Reason Columns (p)", padding=10)
        reason_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(reason_frame, text="These columns are candidate reasons for the conclusion:").pack(anchor=tk.W)
        
        canvas = tk.Canvas(reason_frame, height=150)
        scrollbar = ttk.Scrollbar(reason_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for col in self.columns:
            var = tk.BooleanVar(value=True)
            self.reason_vars[col] = var
            ttk.Checkbutton(scrollable_frame, text=col, variable=var).pack(anchor=tk.W, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15,0))
        
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text="✓ OK", command=self._ok).pack(side=tk.RIGHT, padx=2)
    
    def _on_type_change(self, *args):
        """Ricostruisce widget threshold in base al tipo"""
        for widget in self.threshold_widget_container.winfo_children():
            widget.destroy()
        
        type_str = self.type_var.get()
        
        if type_str == 'Numeric':
            ttk.Label(self.threshold_widget_container, text="Values ≥ this threshold are success:").pack(anchor=tk.W)
            entry = ttk.Entry(self.threshold_widget_container, textvariable=self.threshold_var, width=20)
            entry.pack(anchor=tk.W, pady=(5, 0))
            self.threshold_var.set('70')
        elif type_str == 'Boolean':
            ttk.Label(self.threshold_widget_container, text="Select the value that represents success:").pack(anchor=tk.W)
            bool_vals = ['Yes', 'No', 'True', 'False', '1', '0']
            combo = ttk.Combobox(self.threshold_widget_container, textvariable=self.threshold_var, 
                                values=bool_vals, state='readonly', width=15)
            combo.pack(anchor=tk.W, pady=(5, 0))
            combo.current(0)
        else:  # Categorical
            ttk.Label(self.threshold_widget_container, text="Enter the exact value that represents success:").pack(anchor=tk.W)
            entry = ttk.Entry(self.threshold_widget_container, textvariable=self.threshold_var, width=30)
            entry.pack(anchor=tk.W, pady=(5, 0))
            self.threshold_var.set('high')
    
    def _ok(self):
        """Valida e salva configurazione"""
        try:
            target = self.target_var.get()
            threshold_str = self.threshold_var.get()
            type_str = self.type_var.get()
            
            if not target:
                messagebox.showwarning("Invalid Configuration", "Please select a target column")
                return
            
            if not threshold_str:
                messagebox.showwarning("Invalid Configuration", "Please set a success threshold")
                return
            
            reasons = [col for col, var in self.reason_vars.items() if var.get() and col != target]
            
            if not reasons:
                messagebox.showwarning("Invalid Configuration", "Please select at least one reason column")
                return
            
            col_type = ColumnType.NUMERIC if type_str == 'Numeric' else \
                      ColumnType.BOOLEAN if type_str == 'Boolean' else \
                      ColumnType.CATEGORICAL
            
            if col_type == ColumnType.NUMERIC:
                try:
                    threshold = float(threshold_str)
                except:
                    messagebox.showwarning("Invalid Threshold", "Invalid numeric threshold. Please enter a number.")
                    return
            else:
                threshold = threshold_str
            
            self.result = {
                'target': target,
                'threshold': threshold,
                'col_type': col_type,
                'reasons': reasons
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Error saving configuration:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _cancel(self):
        self.destroy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = MDLApp()
    app.mainloop()


if __name__ == '__main__':
    main()
