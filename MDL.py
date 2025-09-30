# mdl_gui_csv_final.py
# MDL — Ragioni supererogatorie (CSV reale, τ adattiva, S*, marginali post‑S*, tie-breaking deterministico)
# Standard library only: tkinter/ttk, zlib, bz2, csv, textwrap, os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import csv
import random
import zlib
import bz2
import textwrap
import os

# ---------------------------
# Utilità di compressione
# ---------------------------
def bytes_of(s: str) -> bytes:
    return s.encode('utf-8', errors='ignore')

def compressed_len(data: bytes, method: str = 'zlib') -> int:
    if method == 'zlib':
        return len(zlib.compress(data, level=9))
    elif method == 'bz2':
        return len(bz2.compress(data, compresslevel=9))
    else:
        raise ValueError(f"Metodo non supportato: {method}")

# L(x | y) ≈ L(y ⊕ x) - L(y)
def L_cond(x: str, y: str, method: str = 'zlib') -> int:
    sep = "\n§\n"  # separatore deliberato per favorire patterning nei codec
    ly = compressed_len(bytes_of(y), method)
    lyx = compressed_len(bytes_of(y + sep + x), method)
    return max(0, lyx - ly)

# ---------------------------
# Caricamento CSV
# ---------------------------
EXPECTED_HEADERS = [
    "student_id","age","gender","study_hours_per_day","social_media_hours",
    "netflix_hours","part_time_job","attendance_percentage","sleep_hours",
    "diet_quality","exercise_frequency","parental_education_level",
    "internet_quality","mental_health_rating","extracurricular_participation",
    "exam_score"
]

def load_student_habits_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        missing = [h for h in EXPECTED_HEADERS if h not in headers]
        if missing:
            raise ValueError(f"Header CSV mancanti: {missing}")
        for r in rdr:
            rows.append(r)
    return rows

# ---------------------------
# Testi MDL dai dati reali
# ---------------------------
def make_text(core_tokens, other_tokens, core_share: float, total_len: int, rnd: random.Random):
    total_len = max(80, total_len)  # testi più lunghi per robustezza compressiva
    n_core = max(0, min(total_len, int(round(total_len * core_share))))
    toks = [rnd.choice(core_tokens) for _ in range(n_core)] + \
           [rnd.choice(other_tokens) for _ in range(total_len - n_core)]
    rnd.shuffle(toks)
    return " ".join(toks)

def build_texts_from_csv(rows, pass_threshold=70.0, seed=1234):
    rnd = random.Random(seed)

    # Vocabolario di "successo"
    Q = [
        "passo","supero","padronanza","preparato","capito","accurato","solido","chiaro",
        "fluente","forte","stabile","ottimo","completo","robusto","pulito","efficace",
        "sicuro","affidabile","consistente","competente","brillante","abile","capace","determinato"
    ]
    V = Q + [f"x{i}" for i in range(300)]
    NotQ = [t for t in V if t not in Q]

    # Esito pass/fail dal punteggio
    def is_pass(r):
        try:
            return float(r["exam_score"]) >= pass_threshold
        except:
            return False

    # q: descrizione del successo (ricca in Q)
    q_text = make_text(Q, NotQ, 0.90, 900, rnd)

    # Background B + Evidenze E (overlap moderato)
    B_text = make_text(Q, NotQ, 0.20, 420, rnd)
    E_text = make_text(Q, NotQ, 0.15, 300, rnd)

    # Ragioni p dal CSV (7 ragioni, includendo la salute mentale come p7)
    def bool_yes(v): return str(v).strip().lower() in ("yes","y","true","1")
    def cond_p1(r):  # studio intensivo
        try: return float(r["study_hours_per_day"]) >= 4.0
        except: return False
    def cond_p2(r):  # extracurricolari attive
        return bool_yes(r.get("extracurricular_participation","No"))
    def cond_p3(r):  # alta frequenza
        try: return float(r["attendance_percentage"]) >= 90.0
        except: return False
    def cond_p4(r):  # sonno adeguato
        try: return float(r["sleep_hours"]) >= 7.0
        except: return False
    def cond_p5(r):  # esercizio regolare
        try: return int(float(r["exercise_frequency"])) >= 3
        except: return False
    def cond_p6(r):  # internet buona
        return str(r.get("internet_quality","")).strip().lower() == "good"
    def cond_p7(r):  # salute mentale buona
        try: return float(r["mental_health_rating"]) >= 6.0
        except: return False

    conds = [cond_p1, cond_p2, cond_p3, cond_p4, cond_p5, cond_p6, cond_p7]
    base_labels = [
        "p1: studio intensivo",
        "p2: attività extracurricolari",
        "p3: alta frequenza",
        "p4: sonno adeguato",
        "p5: esercizio regolare",
        "p6: internet buona",
        "p7: salute mentale buona"
    ]

    # Pass-rate per tarare l'overlap informativo dei testi delle p
    reasons, labels = [], []
    for name, cond in zip(base_labels, conds):
        subset = [r for r in rows if cond(r)]
        pass_rate = (sum(1 for r in subset if is_pass(r)) / len(subset)) if subset else 0.0
        ovl = max(0.10, min(0.92, 0.22 + 0.70 * pass_rate))  # mapping morbido in [0.10, 0.92]
        rtxt = make_text(Q, NotQ, ovl, 320, rnd)
        reasons.append(rtxt)
        labels.append(f"{name} (ovl≈{ovl:.2f})")

    # Ragione dominante r: studio alto + frequenza altissima + salute mentale buona
    subset_dom = []
    for r in rows:
        try:
            dom = (float(r["study_hours_per_day"]) >= 5.0 and
                   float(r["attendance_percentage"]) >= 95.0 and
                   float(r["mental_health_rating"]) >= 6.0)
        except:
            dom = False
        if dom:
            subset_dom.append(r)
    pass_rate_dom = (sum(1 for r in subset_dom if is_pass(r)) / len(subset_dom)) if subset_dom else 0.0
    ovl_dom = max(0.70, min(0.96, 0.72 + 0.20 * pass_rate_dom))
    r_dom = make_text(Q, NotQ, ovl_dom, 340, rnd)

    return {
        "q": q_text,
        "B": B_text,
        "E": E_text,
        "reasons": reasons,
        "labels": labels,
        "r_dom": r_dom
    }

# ---------------------------
# MDL: Δ̂ base, τ adattiva, selezione e marginali post‑S*
# ---------------------------
def mdl_deltas_base(data, method='zlib', include_dom=False):
    C = data["B"] + "\n" + data["E"]
    if include_dom:
        C = C + "\n" + data["r_dom"]
    L_q_C = L_cond(data["q"], C, method=method)
    deltas = []
    for p in data["reasons"]:
        Cp = C + "\n" + p
        L_q_Cp = L_cond(data["q"], Cp, method=method)
        d_hat = 0.0 if L_q_C == 0 else (L_q_C - L_q_Cp) / float(L_q_C)
        deltas.append(max(-1.0, min(1.0, d_hat)))
    return deltas, L_q_C

def adaptive_tau_from_base_deltas(deltas):
    # Criterio del “gomito” su Δ̂ ordinate desc, con guard-rail
    ds = sorted([max(0.0, min(1.0, d)) for d in deltas], reverse=True)
    if len(ds) < 2:
        return 0.15
    gaps = [ds[i] - ds[i+1] for i in range(len(ds)-1)]
    i_star = max(range(len(gaps)), key=lambda i: gaps[i]) if gaps else 0
    tau = (ds[i_star] + ds[i_star+1]) / 2.0 if i_star < len(ds)-1 else ds[-1]
    return max(0.02, min(0.40, tau))

def greedy_select_mdl(data, method='zlib', tau=0.15, include_dom=False):
    # Selezione con bordo inclusivo (>= tau), EPS numerico, tie-breaking deterministico su indice
    EPS = 1e-12
    C = data["B"] + "\n" + data["E"]
    if include_dom:
        C = C + "\n" + data["r_dom"]
    q = data["q"]
    reasons = data["reasons"]
    n = len(reasons)
    remaining = list(range(n))
    selected = []
    marginal_at_pick = [0.0] * n
    L_current = L_cond(q, C, method=method)

    while True:
        best_i, best_gain = None, 0.0
        for i in remaining:
            Cp = C + "\n" + reasons[i]
            L_q_Cp = L_cond(q, Cp, method=method)
            gain = 0.0 if L_current == 0 else (L_current - L_q_Cp) / float(L_current)
            if (gain > best_gain + EPS) or (abs(gain - best_gain) <= EPS and (best_i is None or i < best_i)):
                best_gain, best_i = gain, i
        if best_i is None or best_gain + EPS < tau:
            break
        selected.append(best_i)
        marginal_at_pick[best_i] = best_gain
        C = C + "\n" + reasons[best_i]
        L_current = L_cond(q, C, method=method)
        remaining.remove(best_i)

    # Marginali post‑S* rispetto a C ∪ S*
    post_C = data["B"] + "\n" + data["E"]
    if include_dom:
        post_C = post_C + "\n" + data["r_dom"]
    for i in selected:
        post_C = post_C + "\n" + reasons[i]
    L_q_post = L_cond(q, post_C, method=method)
    post_marginal = []
    for i in range(n):
        Cp = post_C + "\n" + reasons[i]
        L_q_Cp = L_cond(q, Cp, method=method)
        gm = 0.0 if L_q_post == 0 else (L_q_post - L_q_Cp) / float(L_q_post)
        post_marginal.append(max(-1.0, min(1.0, gm)))
    return selected, marginal_at_pick, post_marginal

# ---------------------------
# App GUI
# ---------------------------
class MDLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MDL — Ragioni supererogatorie (CSV reale, τ adattiva, S*, marginali post‑S*)")
        self.geometry("1280x900")
        self.minsize(1080, 740)

        # Tema ttk con fallback
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass

        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=10)
        self.option_add("*Font", base_font)

        # Stato
        self.rows = []
        self.data = None
        self.method = tk.StringVar(value='zlib')
        self.include_dom = tk.BooleanVar(value=False)  # toggle “r” lasciato opzionale
        self.sort_bars = tk.BooleanVar(value=True)
        self.mdl_run = False
        self.selected = []
        self.marginal_at_pick = []
        self.post_marginal = []
        self.tau_value = None
        self.deltas_base = []
        self.labels_cur = []

        # Notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.page_dash = ttk.Frame(self.nb)
        self.page_data = ttk.Frame(self.nb)
        self.page_help = ttk.Frame(self.nb)
        self.nb.add(self.page_dash, text="Dashboard")
        self.nb.add(self.page_data, text="Dati")
        self.nb.add(self.page_help, text="Spiegazione")

        # Costruzione GUI
        self._build_dashboard()
        self._build_data_tab()
        self._build_help_tab()

        # Carica CSV e prepara tutto
        self._initial_load()

    # ---------- Caricamento iniziale ----------
    def _initial_load(self):
        default_path = "student_habits_performance.csv"
        path = default_path if os.path.exists(default_path) else filedialog.askopenfilename(
            title="Seleziona student_habits_performance.csv",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
        )
        if not path:
            messagebox.showerror("Errore", "Nessun file CSV selezionato.")
            self.destroy()
            return
        try:
            self.rows = load_student_habits_csv(path)
        except Exception as e:
            messagebox.showerror("Errore CSV", str(e))
            self.destroy()
            return

        self._populate_table()

        self.data = build_texts_from_csv(self.rows, pass_threshold=70.0, seed=1234)
        self.labels_cur = self.data["labels"]
        self._recompute_base()

    # ---------- Dashboard ----------
    def _build_dashboard(self):
        top = ttk.Frame(self.page_dash, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Compressore:").pack(side=tk.LEFT)
        self.method_cb = ttk.Combobox(top, values=["zlib","bz2"],
                                      textvariable=self.method, width=6, state="readonly")
        self.method_cb.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.method_cb.pack(side=tk.LEFT, padx=(4,12))

        self.dom_cb = ttk.Checkbutton(top, text="Aggiungi ragione dominante r (contesto C∪r)",
                                      variable=self.include_dom, command=self._on_param_change)
        self.dom_cb.pack(side=tk.LEFT, padx=(0,12))

        self.sort_cb = ttk.Checkbutton(top, text="Ordina barre per Δ̂ base",
                                       variable=self.sort_bars, command=self._redraw)
        self.sort_cb.pack(side=tk.LEFT, padx=(0,12))

        self.run_btn = ttk.Button(top, text="Esegui metodo MDL (τ adattiva)", command=self._run_mdl)
        self.run_btn.pack(side=tk.LEFT)

        # Area centrale
        mid = ttk.Frame(self.page_dash, padding=(10, 6))
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(mid, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        right = ttk.Frame(mid, width=440)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.summary_lbl = ttk.Label(right, text="—", anchor="w", justify="left")
        self.summary_lbl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2,6))

        # Contesto corrente
        self.context_lbl = ttk.Label(right, text="Contesto: C (B+E)")
        self.context_lbl.pack(side=tk.TOP, anchor="w", padx=8, pady=(0,8))

        # Legenda
        leg = ttk.LabelFrame(right, text="Legenda", padding=(8,8))
        leg.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        legend_text = (
            "• Barre: altezza = Δ̂_C(p; q) (riduzione relativa) | etichetta: base | post‑S*\n"
            "• Linea rossa: soglia τ adattiva calcolata al click\n"
            "• Contorno verde + numero: p selezionate (S*) in ordine di scelta\n"
            "• Colore arancione (dopo esecuzione): p supererogatorie rispetto a C ∪ S*"
        )
        ttk.Label(leg, text=textwrap.fill(legend_text, 58), justify="left").pack(side=tk.TOP, anchor="w")

        # Risultato MDL
        res = ttk.LabelFrame(right, text="Risultato MDL", padding=(8,8))
        res.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=8, pady=6)

        self.tau_lbl = ttk.Label(res, text="τ adattiva: —")
        self.tau_lbl.pack(side=tk.TOP, anchor="w", pady=(0,6))

        self.result_sel_title = ttk.Label(res, text="Necessarie per spiegare q (S*):")
        self.result_sel_title.pack(side=tk.TOP, anchor="w")
        self.result_sel_lbl = ttk.Label(res, text="— (premere 'Esegui metodo MDL')", justify="left")
        self.result_sel_lbl.pack(side=tk.TOP, anchor="w", pady=(0,8))

        self.result_sup_title = ttk.Label(res, text="Ragioni supererogatorie (rispetto a C ∪ S*):")
        self.result_sup_title.pack(side=tk.TOP, anchor="w")
        self.result_sup_lbl = ttk.Label(res, text="—", justify="left")
        self.result_sup_lbl.pack(side=tk.TOP, anchor="w")

        # Info compressori
        comp = ttk.LabelFrame(right, text="Compressori (zlib vs bz2)", padding=(8,8))
        comp.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        comp_info = ("zlib = veloce, buon rapporto; bz2 = più compressione ma più lento; "
                     "sono proxy di complessità per stimare L(·) e Δ̂.")
        ttk.Label(comp, text=textwrap.fill(comp_info, 58), justify="left").pack(side=tk.TOP, anchor="w")

        # Barra di stato inferiore
        bottom = ttk.Frame(self.page_dash, padding=(10, 6))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_lbl = ttk.Label(bottom, text="Pronto")
        self.status_lbl.pack(side=tk.LEFT)

    # ---------- Tabella Dati ----------
    def _build_data_tab(self):
        frame = ttk.Frame(self.page_data, padding=(10,8))
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("student_id","age","gender","study_hours_per_day","attendance_percentage",
                "sleep_hours","exercise_frequency","mental_health_rating",
                "extracurricular_participation","internet_quality","exam_score")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        headings = {
            "student_id":"ID", "age":"Età", "gender":"Genere", "study_hours_per_day":"Ore studio",
            "attendance_percentage":"Frequenza %", "sleep_hours":"Ore sonno",
            "exercise_frequency":"Esercizi/sett.", "mental_health_rating":"Salute mentale",
            "extracurricular_participation":"Extracurr.", "internet_quality":"Internet", "exam_score":"Punteggio esame"
        }
        widths = {
            "student_id":90, "age":60, "gender":80, "study_hours_per_day":100,
            "attendance_percentage":110, "sleep_hours":95, "exercise_frequency":110,
            "mental_health_rating":120, "extracurricular_participation":100, "internet_quality":90, "exam_score":120
        }
        for c in cols:
            self.tree.heading(c, text=headings[c], command=lambda col=c: self._sort_by(col, False))
            self.tree.column(c, width=widths[c], anchor="center")

        self.tree.tag_configure("odd", background="#f7f9ff")
        self.tree.tag_configure("even", background="#ffffff")

    def _populate_table(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            vals = (
                r.get("student_id",""),
                r.get("age",""),
                r.get("gender",""),
                r.get("study_hours_per_day",""),
                r.get("attendance_percentage",""),
                r.get("sleep_hours",""),
                r.get("exercise_frequency",""),
                r.get("mental_health_rating",""),
                r.get("extracurricular_participation",""),
                r.get("internet_quality",""),
                r.get("exam_score",""),
            )
            tag = "odd" if i % 2 else "even"
            self.tree.insert("", "end", values=vals, tags=(tag,))

    def _sort_by(self, col, descending):
        cols = self.tree["columns"]
        col_idx = cols.index(col)
        data = []
        for iid in self.tree.get_children(""):
            vals = self.tree.item(iid, "values")
            data.append((vals[col_idx], vals, iid))
        def conv(v):
            try:
                return float(str(v).replace(",", "."))
            except:
                return str(v)
        data.sort(key=lambda t: conv(t[0]), reverse=descending)
        for idx, (_, vals, iid) in enumerate(data):
            self.tree.move(iid, "", idx)
        self.tree.heading(col, command=lambda c=col: self._sort_by(c, not descending))

    # ---------- Spiegazione ----------
    def _build_help_tab(self):
        frame = ttk.Frame(self.page_help, padding=(12,10))
        frame.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(frame, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(state=tk.NORMAL)
        explanation = """
Obiettivo
Selezionare un set minimo S* di ragioni che spiegano q nel contesto C secondo MDL; le altre sono supererogatorie rispetto a C ∪ S*.

Definizioni operative
- Δ̂_C(p; q) = (L(q|C) − L(q|C,p)) / L(q|C), con L(x|y) ≈ L(y⊕x) − L(y).
- τ adattiva: stimata dal profilo di Δ̂ al momento del click (criterio del “gomito”), vincolata in [0.02, 0.40].
- Selezione avara con bordo inclusivo (Δ̂ ≥ τ) e tie‑breaking deterministico.
- Marginali post‑S*: calcolati rispetto a C ∪ S* per evidenziare supererogatorietà nel contesto finale.
- Ragione dominante r: opzionale per mostrare la non‑monotonicità (C → C ∪ r).

Interpretazione
- Contorno verde + numero: p ∈ S* (ordine di selezione).
- Etichetta barra: base | post‑S* per confrontare contributo isolato e marginale finale.
- Leggere sempre il contesto indicato (C oppure C ∪ r) quando si confrontano i risultati.
        """.strip()
        txt.insert("1.0", explanation)
        txt.configure(state=tk.DISABLED)

    # ---------- Logica ----------
    def _on_param_change(self):
        # Cambiando compressore o r si invalida l'esecuzione precedente
        self.mdl_run = False
        self.selected = []
        self.marginal_at_pick = []
        self.post_marginal = []
        self.tau_value = None
        self._recompute_base()

    def _recompute_base(self):
        if not self.data:
            return
        self.deltas_base, L_q_C = mdl_deltas_base(
            self.data, method=self.method.get(), include_dom=self.include_dom.get()
        )
        dom = "C ∪ r" if self.include_dom.get() else "C (B+E)"
        self.context_lbl.config(text=f"Contesto: {dom}")
        status = "MDL non ancora eseguito — premere 'Esegui metodo MDL' (τ sarà calcolata)"
        self.summary_lbl.config(
            text=f"{status} | comp = {self.method.get()} | contesto = {dom} | L(q|C) = {L_q_C}"
        )
        self.tau_lbl.config(text="τ adattiva: —")
        self.result_sel_lbl.config(text="— (premere 'Esegui metodo MDL')")
        self.result_sup_lbl.config(text="—")
        self.status_lbl.config(text="Pronto")
        self._redraw()

    def _run_mdl(self):
        if not self.data:
            return
        self.status_lbl.config(text="Esecuzione MDL in corso…")
        self.update_idletasks()

        # 1) τ adattiva dai Δ̂ base
        tau = adaptive_tau_from_base_deltas(self.deltas_base)
        self.tau_value = tau
        self.tau_lbl.config(text=f"τ adattiva: {tau:.3f}")

        # 2) selezione congiunta + marginali post‑S*
        sel, marginal_at_pick, post_marginal = greedy_select_mdl(
            self.data, method=self.method.get(), tau=tau, include_dom=self.include_dom.get()
        )
        self.selected = sel
        self.marginal_at_pick = marginal_at_pick
        self.post_marginal = post_marginal
        self.mdl_run = True

        # 3) risultati testuali
        labels = self.data["labels"]
        if self.selected:
            sel_lines = [f"• {labels[i].split(' (')[0]} (Δ̂ marg ≈ {self.marginal_at_pick[i]*100:.0f}%)" for i in self.selected]
        else:
            sel_lines = ["• Nessuna (tutte sotto τ)"]
        sup_lines = [f"• {labels[i].split(' (')[0]}" for i in range(len(labels)) if i not in self.selected]
        self.result_sel_lbl.config(text="\n".join(sel_lines))
        self.result_sup_lbl.config(text="\n".join(sup_lines))

        n_total = len(labels)
        n_sel = len(self.selected)
        n_sup = n_total - n_sel
        dom = "C ∪ r" if self.include_dom.get() else "C (B+E)"
        self.summary_lbl.config(
            text=f"selezionate: {n_sel}/{n_total} | supererogatorie: {n_sup}/{n_total} | τ(adattiva) = {tau:.3f} | comp = {self.method.get()} | contesto = {dom}"
        )
        self.status_lbl.config(text="MDL eseguito")
        self._redraw()

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        if not self.deltas_base:
            return

        W = c.winfo_width() or 1000
        H = c.winfo_height() or 600
        pad_l, pad_r, pad_t, pad_b = 120, 30, 60, 110

        # Ordina per Δ̂ base, se richiesto
        order = list(range(len(self.deltas_base)))
        if self.sort_bars.get():
            order.sort(key=lambda i: self.deltas_base[i], reverse=True)

        # Assi
        c.create_line(pad_l, H - pad_b, W - pad_r, H - pad_b, fill="#333", width=1)
        c.create_line(pad_l, H - pad_b, pad_l, pad_t, fill="#333", width=1)

        # Griglia Y
        for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = (H - pad_b) - yv * (H - pad_t - pad_b)
            c.create_line(pad_l, y, W - pad_r, y, fill="#e9e9e9")
            c.create_text(pad_l-55, y, text=f"{yv:.2f}", anchor="e", fill="#555")

        # Linea τ se MDL eseguito
        if self.mdl_run and self.tau_value is not None:
            tau = self.tau_value
            y_tau = (H - pad_b) - tau * (H - pad_t - pad_b)
            c.create_line(pad_l, y_tau, W - pad_r, y_tau, fill="red", width=2, dash=(5,4))
            c.create_text(W - pad_r, y_tau - 12, text=f"τ={tau:.2f}", anchor="e", fill="red")

        labels = [self.data["labels"][i].split(" (")[0] for i in order]
        deltas = [max(0.0, min(1.0, self.deltas_base[i])) for i in order]

        # Post‑S* marginal: per selezionate mostra il gain al pick, altrimenti il marg. post‑S*
        post = [self.post_marginal[i] if self.post_marginal else 0.0 for i in order]
        pick = [self.marginal_at_pick[i] if self.marginal_at_pick else 0.0 for i in order]

        n = len(deltas)
        plot_w = (W - pad_l - pad_r)
        bar_w = max(28, plot_w / max(14, 2*n))

        for rank, i in enumerate(order, start=1):
            d = deltas[rank-1]
            pm = (pick[rank-1] if self.mdl_run and i in self.selected else post[rank-1])
            pm = max(0.0, min(1.0, pm))

            x_center = pad_l + (rank - 0.5) * (plot_w / n)
            x0 = x_center - bar_w/2
            x1 = x_center + bar_w/2
            y1 = (H - pad_b)
            y0 = y1 - d * (H - pad_t - pad_b)

            rect = c.create_rectangle(x0, y0, x1, y1, fill="#3367CC", outline="white", width=1)

            if self.mdl_run:
                if i in self.selected:
                    c.create_rectangle(x0, y0, x1, y1, outline="#2E8B57", width=3)
                    order_num = self.selected.index(i) + 1
                    c.create_oval(x0+4, y0-24, x0+28, y0-4, fill="#2E8B57", outline="")
                    c.create_text(x0+16, y0-14, text=str(order_num), fill="white", font=("TkDefaultFont", 10, "bold"))
                else:
                    c.itemconfig(rect, fill="#F18F34")

            # Etichetta percentuale doppia: base | post‑S*
            c.create_text(x_center, y0 - 18, text=f"{d*100:.0f}% | {pm*100:.0f}%", fill="#333", font=("TkDefaultFont", 11, "bold"))
            # Etichetta X (wrapping)
            label_text = labels[rank-1] if len(labels[rank-1]) <= 18 else "\n".join(textwrap.wrap(labels[rank-1], width=18))
            c.create_text(x_center, H - pad_b + 34, text=label_text, anchor="n", fill="#444")

        # Titoli
        c.create_text((pad_l + W - pad_r)//2, 24, text="Riduzione relativa Δ̂_C(p; q) — verde: selezionate (S*), arancione: supererogatorie", fill="#333")
        c.create_text(34, (pad_t + H - pad_b)//2, text="Δ̂_C(p; q)", fill="#333", angle=90)
        c.create_text((pad_l + W - pad_r)//2, H - 48, text="Ragioni", fill="#333")


if __name__ == "__main__":
    app = MDLApp()
    app.mainloop()
