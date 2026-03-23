"""
Glaser-Verfahren (Glaser Method) – Bauphysikalische Taupunktanalyse
====================================================================
Berechnet Temperaturverlauf, Dampfdruckverlauf und Sättigungsdampfdruck
durch ein mehrschichtiges Bauteil nach DIN 4108-3 / ISO 13788.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy import units as u
import io 

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glaser-Verfahren",
    page_icon="🏗️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
  code, .stCode { font-family: 'DM Mono', monospace; }

  .stApp { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

  .layer-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
  }
  .warning-card {
    background: #2d1b0e; border: 1px solid #e05d00;
    border-radius: 8px; padding: 0.8rem 1.2rem; margin-top: 0.5rem;
  }
  .ok-card {
    background: #0d2818; border: 1px solid #2ea043;
    border-radius: 8px; padding: 0.8rem 1.2rem; margin-top: 0.5rem;
  }
  [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; font-size: 1.4rem; }
  .stButton > button {
    background: #238636; color: #fff; border: none;
    border-radius: 6px; font-family: 'Syne', sans-serif; font-weight: 600;
  }
  .stButton > button:hover { background: #2ea043; }
  input[type="number"] { font-family: 'DM Mono', monospace !important; }
  .section-head {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #8b949e;
    border-bottom: 1px solid #30363d; padding-bottom: 4px; margin: 1.5rem 0 1rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSIKALISCHE KLASSEN
# ═══════════════════════════════════════════════════════════════════════════════
class Layer:
    def __init__(self, name, thickness, heat_conductivity, vapor_diffusion_resistance_factor,vapor_diffusion_coeff=2e10 *u.kg/(u.m*u.s*u.Pa)):
        """
        Schicht im Bauteilaufbau.

        Params
        ----------
        name                              : Bezeichnung der Schicht
        thickness                         : Dicke d  [m]
        heat_conductivity                 : Wärmeleitfähigkeit λ  [W/(m·K)]
        vapor_diffusion_resistance_factor : Diffusionswiderstandszahl μ  [–]
        """
        self.name = name
        self.thickness = thickness
        self.heat_conductivity = heat_conductivity
        self.vapor_diffusion_resistance_factor = vapor_diffusion_resistance_factor
        self.vapor_diffusion_coeff    = vapor_diffusion_coeff
    def compute_heat_resistance(self):
        """
        Wärmedurchgangswiderstand der Schicht:
            R = d / λ   [m²·K/W]
        """
        return (self.thickness / self.heat_conductivity).to(u.m**2 * u.K / u.W)
    
    def compute_diffusion_resitance(self):
            
        d    = self.thickness 
        mu   = self.vapor_diffusion_resistance_factor 
        delta = self.vapor_diffusion_coeff 

        Z = (mu * d) / delta 

        return Z.to(u.m**2*u.s*u.Pa/u.kg)


class Glaser:
    def __init__(self, layers, T_in, T_ext, R_si, R_se, phi_in, phi_ext):
        """
        Führt die Glaser-Berechnung für einen mehrschichtigen Bauteilaufbau durch.

        Parameters
        ----------
        layers  : Liste von Layer-Objekten (von außen nach innen)
        T_in    : Innentemperatur  [K]
        T_ext   : Außentemperatur  [K]
        R_si    : Innerer Oberflächenwiderstand  [m²·K/W]
        R_se    : Äußerer Oberflächenwiderstand  [m²·K/W]
        phi_in  : Relative Innenfeuchte   [dimensionslos, 0–1]
        phi_ext : Relative Außenfeuchte   [dimensionslos, 0–1]
        """
        self.layers     = layers
        self.T_in       = T_in
        self.T_ext      = T_ext
        self.R_si       = R_si
        self.R_se       = R_se
        self.phi_in     = phi_in
        self.phi_ext    = phi_ext
        self.delta_T    = T_in - T_ext
        self.num_layers = len(layers)

    def compute_total_resistance(self):
        """
        Gesamtwärmedurchgangswiderstand des Bauteils:
            R_ges = R_si + Σ R_i + R_se   [m²·K/W]

        Beinhaltet die Oberflächenwiderstände beider Seiten.
        """
        R_layers = sum(
            layer.compute_heat_resistance().to(u.m**2 * u.K / u.W).value
            for layer in self.layers
        )
        return (self.R_si + R_layers * u.m**2 * u.K / u.W + self.R_se).to(u.m**2 * u.K / u.W)

    def compute_surface_temperatures(self):
        """
        Temperaturverlauf an den n+1 Schichtgrenzen (stationäre Wärmeleitung).

        Der stationäre Wärmestrom ist über das gesamte Bauteil konstant:
            q = ΔT / R_ges   [W/m²]

        Die Temperatur fällt über jeden Teilwiderstand R_j proportional ab:
            ΔT_j = ΔT · R_j / R_ges

        Iteration von T_ext nach innen; Reihenfolge der Widerstände:
            R_si → R_1 … R_n (Schichten) → R_se
        """
        R_total = self.compute_total_resistance()
        T = self.T_ext
        T_list = [T.to(u.K).value]

        for ii in range(self.num_layers + 2):
            if 0 < ii <= self.num_layers:
                layer = self.layers[ii - 1]
                R = layer.thickness / layer.heat_conductivity
            elif ii == 0:
                R = self.R_si
            else:
                R = self.R_se
            T = T + R / R_total * self.delta_T
            T_list.append(T.to(u.K).value)

        return np.array(T_list) * u.K

    def magnus_formula(self, T):
        """
        Sättigungsdampfdruck nach der Magnus-Formel:
            p_sat = 6.112 hPa · exp( 22.46 · T / (272.62°C + T) )

        T muss als astropy-Größe in °C übergeben werden.
        Gültig für ca. −40 °C … +60 °C.
        """
        return (6.112 * u.hPa * np.exp(22.46 * T / (272.62 * u.Celsius + T))).to(u.Pa)

    def compute_saturation_vapor_pressure(self):
        """
        Sättigungsdampfdruck p_sat [Pa] an jeder der n+1 Schichtgrenzen,
        berechnet aus der jeweiligen Grenzflächentemperatur mittels Magnus-Formel.
        """
        T_K = self.compute_surface_temperatures()
        T_C = T_K.to(u.Celsius, equivalencies=u.temperature())
        p_sat = np.array([self.magnus_formula(T).to(u.Pa).value for T in T_C])
        return p_sat * u.Pa

    def compute_vapor_pressures(self):
        """
        Tatsächlicher Dampfdruckverlauf durch das Bauteil.

        Randbedingungen aus Klima und Sättigungsdampfdruck an den Oberflächen:
            p_ext = φ_ext · p_sat(T_ext)
            p_in  = φ_in  · p_sat(T_in)

        Dampfdruck linear in x zwischen p_ext und p_in interpoliert 
        – gleichmäßig über alle n+1 Grenzflächen,
         TO DO: abh. vom μ-Wert der Schichten machen....

            p_j = p_ext + j/n · (p_in - p_ext),   j = 0 … n
        """
        p_sat = self.compute_saturation_vapor_pressure()
        p_ext = self.phi_ext * p_sat[0]
        p_in  = self.phi_in  * p_sat[-1]

        n = self.num_layers
        #p_vapor = np.linspace(p_ext.value,p_in.value,n+3)*u.Pa  #np.array([p_ext.value + j / n * (p_in.value - p_ext.value)  for j in range(n + 3)]) * u.Pa

        return  p_ext, p_in

    def compute_results(self):
        """
        Aggregiert alle Ergebnisgrößen der Glaser-Auswertung:
            T_array     : Temperaturen an Schichtgrenzen  [K]
            p_sat_array : Sättigungsdampfdrücke           [Pa]
            p_vapor     : tatsächlicher Dampfdruckverlauf [Pa]
            p_ext       : Dampfdruck außen                [Pa]
            p_in        : Dampfdruck innen                [Pa]
        """
        T_array     = self.compute_surface_temperatures()
        p_sat_array = self.compute_saturation_vapor_pressure()
        p_ext, p_in = self.compute_vapor_pressures()
        return T_array, p_sat_array, p_ext, p_in


def check_condensation(vapor_pressures, sat_pressures):
    """
    Taupunktprüfung an jeder Schichtgrenze:
        Kondensation wenn p(x) > p_sat(T(x))

    Return:
        Liste von Booleans (True = Tauwasserausfall möglich).
    """
    return [float(p) > float(ps) for p, ps in zip(vapor_pressures, sat_pressures)]


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE – Standard-Schichten
# ═══════════════════════════════════════════════════════════════════════════════

if "layers" not in st.session_state:
    # [Name, Dicke [mm], λ [W/mK], μ [–]]
    st.session_state.layers = [
        ["Außenputz",     20,  0.87,  10],
        ["Mineralwolle",  120,  0.045, 3 ],
        ["Kalksandstein", 240,  0.56,  10],
        ["Innenputz",     15, 0.87,  10],
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Klimabedingungen
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌡️ Klimabedingungen")
    st.markdown('<div class="section-head">Außenklima</div>', unsafe_allow_html=True)
    T_outside_C = st.number_input("Außentemperatur [°C]", value=-10.0, step=0.5, format="%.1f")
    rH_outside  = st.number_input("Relative Feuchte außen [%]", value=80.0,
                                   min_value=0.0, max_value=100.0, step=1.0, format="%.0f")

    st.markdown('<div class="section-head">Innenklima</div>', unsafe_allow_html=True)
    T_inside_C = st.number_input("Innentemperatur [°C]", value=20.0, step=0.5, format="%.1f")
    rH_inside  = st.number_input("Relative Feuchte innen [%]", value=50.0,
                                  min_value=0.0, max_value=100.0, step=1.0, format="%.0f")

    st.markdown('<div class="section-head">Innenklima</div>', unsafe_allow_html=True)
    R_se  = st.number_input("R_se Oberflächenwiderstand (s. DIN 4108-4) [m2K/W]", value=0.04,min_value=0.0, max_value=10.0, step=0.001, format="%.3f")
    R_si  = st.number_input("R_si Oberflächenwiderstand (s. DIN 4108-4) [m2K/W]", value=0.13,min_value=0.0, max_value=10.0, step=0.001, format="%.3f")


    st.markdown("---")
    st.markdown("**Hinweis:** Oberflächenwiderstände nach :  \nR_si = 0.13, R_se = 0.04 m²K/W")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN – Schichtaufbau
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# Glaser-Verfahren")
st.markdown("*Bauphysikalische Analyse des Tauwasserausfalls in mehrschichtigen Bauteilen*")

st.markdown('<div class="section-head">Schichtaufbau — von außen nach innen</div>',
            unsafe_allow_html=True)

col_add, col_del, _ = st.columns([1, 1, 4])
with col_add:
    if st.button("＋ Neue Schicht hinzufügen"):
        st.session_state.layers.append(["Neue Schicht", 100, 0.50, 5])
        st.rerun()
with col_del:
    if st.button("－ Letzte Schicht entfernen") and len(st.session_state.layers) > 1:
        st.session_state.layers.pop()
        st.rerun()

col_inputs, col_plot = st.columns([1.1, 1.5])   # Input | Plot
with col_inputs :
    for i, layer in enumerate(st.session_state.layers):
        #st.markdown('<div class="layer-card">', unsafe_allow_html=True)
        cols = st.columns([1.5, 1., 1., 1.])
        with cols[0]:
            st.session_state.layers[i][0] = st.text_input(f"Bezeichnung Schicht", value=layer[0], key=f"name_{i}")
        with cols[1]:
            st.session_state.layers[i][1] = st.number_input("Schichtdicke d [mm]", value=float(layer[1]), min_value=1., step=1., format="%.0f", key=f"d_{i}")
        with cols[2]:
            st.session_state.layers[i][2] = st.number_input("Wärmeleitf. λ [W/mK]", value=float(layer[2]), min_value=0.001, step=0.01, format="%.3f", key=f"lam_{i}")
        with cols[3]:
            st.session_state.layers[i][3] = st.number_input("μ-Wert [–]", value=float(layer[3]), min_value=1.0, step=1.0, format="%.0f", key=f"mu_{i}")
        #st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BERECHNUNG
# ═══════════════════════════════════════════════════════════════════════════════

# Astropy-Layer-Objekte aus Session-State erzeugen
layer_objects = [Layer(row[0], row[1] * u.mm, row[2] * u.W / u.m / u.K, row[3] * u.one)  for row in st.session_state.layers]

# Klima-Einheiten
T_ext_K = (T_outside_C + 273.15) * u.K
T_in_K  = (T_inside_C  + 273.15) * u.K
R_si    = R_si * u.m**2 * u.K / u.W #0.13 * u.m**2 * u.K / u.W
R_se    = R_se * u.m**2 * u.K / u.W #0.04 * u.m**2 * u.K / u.W
phi_ext = (rH_outside / 100.0) * u.one
phi_in  = (rH_inside  / 100.0) * u.one

glaser = Glaser(layer_objects, T_in_K, T_ext_K, R_si, R_se, phi_in, phi_ext)
T_array, p_sat_array, p_ext, p_in = glaser.compute_results()

mu_arr = np.array([row[3] for row in st.session_state.layers])


# Schichtgrenzen: n+3 Punkte, x = 0 … d_gesamt
wall_x = [0.0]
for row in st.session_state.layers:
    wall_x.append(wall_x[-1] + row[1])

total_thickness = sum(row[1] for row in st.session_state.layers)
offset = total_thickness * 0.2   # virtueller Abstand der Luftpunkte im Plot [mm]
x_arr  = np.array([-offset] + wall_x + [wall_x[-1] + offset])         # x für Temperaturkurve: n+3 Punkte  (Luftpunkt außen + Wand + Luftpunkt innen)
x_wall = np.array(wall_x)          

p_vapor = p_ext +  x_arr*(p_in-p_ext)/x_arr[-1]


# In plain Python-Listen für UI und Plotting umwandeln
temperatures    = T_array.to(u.Celsius, equivalencies=u.temperature()).value.tolist()
sat_pressures   = p_sat_array.value.tolist()
vapor_pressures = p_vapor.value.tolist()

condensation_flags = check_condensation(vapor_pressures, sat_pressures)
print(condensation_flags)

R_total = glaser.compute_total_resistance()
U_value = 1.0 / R_total.value   # U-Wert  [W/(m²K)]
q_heat  = (T_in_K.value - T_ext_K.value) / R_total.value  # Wärmestrom  [W/m²]

# ═══════════════════════════════════════════════════════════════════════════════
# ERGEBNISSE – Kennwerte
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Ergebnisse</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("U-Wert",       f"{U_value:.3f} W/m²K")
m2.metric(r"Wärmestrom $Q$", f"{q_heat:.1f} W/m²")
m3.metric(r"$p_{\rm sat}$ außen",  f"{sat_pressures[0]:.0f} Pa")
m4.metric(r"$p_{\rm sat}$ innen",  f"{sat_pressures[-1]:.0f} Pa")

# ═══════════════════════════════════════════════════════════════════════════════
# ERGEBNISTABELLE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Grenzflächen-Tabelle</div>', unsafe_allow_html=True)

layer_names = [row[0] for row in st.session_state.layers]
boundary_labels = (
    ["Außenklima",f"Außenoberfläche | {layer_names[0]}" ]
    + [f"{layer_names[i]} | {layer_names[i+1]}" for i in range(len(layer_names) - 1)]
    + [f"{layer_names[-1]} | Innenoberfläche", "Innenklima"]
)

import pandas as pd
rows = []
for j, label in enumerate(boundary_labels):
    cond = condensation_flags[j]
    rows.append({
        "Grenzfläche":     label,
        "Temperatur [°C]": f"{temperatures[j]:.2f}",
        "p_Dampf [Pa]":    f"{vapor_pressures[j]:.1f}",
        "p_sat [Pa]":      f"{sat_pressures[j]:.1f}",
        "p / p_sat":       f"{vapor_pressures[j] / sat_pressures[j]:.3f}",
        "Kondensation":    "⚠️ JA" if cond else "✅ nein",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if any(condensation_flags):
    st.markdown('<div class="warning-card">⚠️ <strong>Tauwasserausfall festgestellt!</strong> '
                'An mindestens einer Grenzfläche übersteigt der Dampfdruck den Sättigungsdampfdruck. '
                'Kondensation möglich.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="ok-card">✅ <strong>Kein Tauwasserausfall.</strong> '
                'Der Dampfdruck liegt an allen Grenzflächen unter dem Sättigungsdampfdruck.</div>',
                unsafe_allow_html=True)

st.markdown("---")
# Excel-Export
df_export = pd.DataFrame(rows)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="Glaser-Ergebnisse")

st.download_button(
    label="📥 Tabelle als Excel exportieren",
    data=buffer.getvalue(),
    file_name="glaser_ergebnisse.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAMME
# ═══════════════════════════════════════════════════════════════════════════════

#st.markdown('<div class="section-head">Diagramme</div>', unsafe_allow_html=True)

# X-Positionen aufbauen
#
# compute_surface_temperatures() liefert n+3 Einträge für n Schichten:
#   T[0]       : Außenluft (T_ext) – vor der Wand, im Freien
#   T[1]       : Außenwand-Oberfläche (nach R_si)
#   T[2..n]    : Schichtgrenzen
#   T[n+1]     : Innenwand-Oberfläche (nach letzter Schicht)
#   T[n+2]     : Innenluft (T_in) – hinter der Wand, im Raum  (nach R_se)
#
# Vapor/sat arrays haben nur n+1 Einträge (nur Wandgrenzen, keine Luftpunkte).
# Für das Plotting wird deshalb ein separates x_wall für p/p_sat verwendet.


# Schichtgrenzen: n+1 Punkte, x = 0 … d_gesamt
wall_x = [0.0]
for row in st.session_state.layers:
    wall_x.append(wall_x[-1] + row[1])


#x_arr  = np.array([-offset] + wall_x + [wall_x[-1] + offset])         # x für Temperaturkurve: n+3 Punkte  (Luftpunkt außen + Wand + Luftpunkt innen)
#x_wall = np.array(wall_x)                                             # x für Ticks: pos. der Schichten

T_arr  = np.array(temperatures)     # n+3 Werte
p_arr  = np.array(vapor_pressures)  # n+3 Werte
ps_arr = np.array(sat_pressures)    # n+3 Werte

# x_interfaces für axvspan/axvline (echte Wandgrenzen)
x_interfaces = wall_x
x_mid  = [(x_interfaces[i] + x_interfaces[i+1]) / 2
          for i in range(len(st.session_state.layers))]

layer_colors = ["#1f4e79", "#2e7d32", "#6d4c41", "#f57f17",
                "#4a148c", "#880e4f", "#01579b", "#1b5e20"]

fig = plt.figure(figsize=(13, 8), facecolor="#0d1117")
gs  = GridSpec(2, 1, figure=fig, hspace=0.45)

# ── Plot 1: Temperaturverlauf ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#161b22")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363d")
ax1.tick_params(colors="#8b949e", labelsize=9)
ax1.xaxis.label.set_color("#8b949e")
ax1.yaxis.label.set_color("#8b949e")
ax1.set_title("Temperaturverlauf", color="#e6edf3", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Position [mm]", fontsize=9)
ax1.set_ylabel("Temperatur [°C]", fontsize=9)
ax1.grid(True, color="#21262d", linewidth=0.8, linestyle="--")

ax1.set_xticks(x_wall)
ax1.set_xticklabels(np.round(x_wall).astype(int))


for i, (row, color) in enumerate(zip(st.session_state.layers, layer_colors)):
    ax1.axvspan(x_interfaces[i], x_interfaces[i+1], alpha=0.15, color=color, label=row[0])
    #ax1.text(x_mid[i], T_arr.min() - 0.3, row[0], ha="center", va="top",
    #         fontsize=7.5, color="#8b949e",
    #         rotation=90 if row[1] < 0.04 else 0)

for x in x_interfaces:
    ax1.axvline(x, color="#30363d", linewidth=0.8, linestyle=":")

ax1.plot(x_arr, T_arr, color="#58a6ff", linewidth=2.5, marker="o", markersize=6, markerfacecolor="#0d1117", markeredgecolor="#58a6ff", markeredgewidth=2, zorder=5)
#ax1.annotate(r"$T_e$"+f" = {T_outside_C}°C", xy=(x_arr[0],  T_arr[0]+3), xytext=(x_arr[0]  + offset*0.05, T_arr[0]  + 3), color="#58a6ff", fontsize=8)
#ax1.annotate(r"$T_i$"+f" = {T_inside_C}°C",  xy=(x_arr[-1], T_arr[-1]-3), xytext=(x_arr[-1] - offset*0.6,  T_arr[-1] -3), color="#58a6ff", fontsize=8)

ax1.set_xlim(x_arr[0]-5, x_arr[-1]+5)
ax1.set_ylim(T_outside_C-1, T_inside_C+1)

ax1.legend(loc="upper left", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", ncol=1)#len(st.session_state.layers)+1)

# ── Plot 2: Dampfdruckverlauf ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#161b22")
for spine in ax2.spines.values():
    spine.set_edgecolor("#30363d")
ax2.tick_params(colors="#8b949e", labelsize=9)
ax2.xaxis.label.set_color("#8b949e")
ax2.yaxis.label.set_color("#8b949e")
ax2.set_title("Dampfdruck- und Sättigungsdampfdruckverlauf", color="#e6edf3", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Position [mm]", fontsize=9)
ax2.set_ylabel("Druck [Pa]", fontsize=9)
ax2.grid(True, color="#21262d", linewidth=0.8, linestyle="--")

ax2.set_xticks(x_wall)
ax2.set_xticklabels(np.round(x_wall).astype(int))


for i, (row, color) in enumerate(zip(st.session_state.layers, layer_colors)):
    ax2.axvspan(x_interfaces[i], x_interfaces[i+1], alpha=0.15, color=color)
for x in x_interfaces:
    ax2.axvline(x, color="#30363d", linewidth=0.8, linestyle=":")

ax2.plot(x_arr, ps_arr, color="#f0883e", linewidth=2.2, linestyle="--", marker="s", markersize=5, markerfacecolor="#0d1117", markeredgecolor="#f0883e", markeredgewidth=2, label="p_sat(T) – Sättigungsdampfdruck")
ax2.plot(x_arr, p_arr, color="#79c0ff", linewidth=2.5, marker="o", markersize=6, markerfacecolor="#0d1117", markeredgecolor="#79c0ff", markeredgewidth=2, zorder=5, label="p – Dampfdruck (linear interpoliert)")

# Kondensationspunkte
first_cond = True
for j, (cond, x, p) in enumerate(zip(condensation_flags, x_arr, p_arr)):
    if cond:
        ax2.scatter([x], [p], color="#ff7b72", s=120, zorder=8, marker="X", linewidths=1, label="⚠ Kondensation" if first_cond else "")
        #ax2.annotate("Tauwasser!", xy=(x, p), xytext=(x + 5, p + 20), color="#ff7b72", fontsize=8, fontweight="bold")
        first_cond = False

# Kondensationszone füllen
#x_fine  = np.linspace(x_wall[0], x_wall[-1], 500)
#p_fine  = np.interp(x_fine, x_wall, p_arr)
#ps_fine = np.interp(x_fine, x_arr, ps_arr)
#ax2.fill_between(x_fine, p_fine, ps_fine, where=(p_fine > ps_fine),  alpha=0.25, color="#ff7b72", label="Kondensationszone")

#ax2.set_xlim(x_wall[0] - 5, x_wall[-1] + 5)
handles, labels_leg = ax2.get_legend_handles_labels()
seen, unique = set(), []
for h, l in zip(handles, labels_leg):
    if l not in seen:
        seen.add(l)
        unique.append((h, l))
        
ax2.legend(*zip(*unique), loc="lower left",bbox_to_anchor=(0,1), fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
fig.tight_layout(pad=1.5)

with col_plot:
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FUSSZEILE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<small style='color:#8b949e'>Berechnung nach <strong>DIN 4108-3 / ISO 13788</strong> · "
    "Magnus-Formel für Sättigungsdampfdruck · "
    "Dampfdruckverlauf: lineare Interpolation in x · "
    "Stationäre Betrachtung (Winterfall)</small>",
    unsafe_allow_html=True
)