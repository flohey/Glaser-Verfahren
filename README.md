# 🏗️ Glaser-Verfahren – Bauphysikalische Taupunktanalyse

Eine interaktive Streamlit-App zur Berechnung des Tauwasserausfalls in mehrschichtigen Bauteilen nach **DIN 4108-3 / ISO 13788**.

## Was ist das Glaser-Verfahren?

Das Glaser-Verfahren ist eine vereinfachte stationäre Methode zur Beurteilung des Tauwasserausfalls in Bauteilen. Es berechnet den Temperaturverlauf und den Wasserdampfdruckverlauf durch einen mehrschichtigen Wandaufbau und prüft, ob der tatsächliche Dampfdruck an einer Schichtgrenze den Sättigungsdampfdruck überschreitet – in diesem Fall kommt es zur Kondensation. Die App eignet sich besonders für Energieberater, um schnell eine erste Abschätzung des hygrothermischen Verhaltens eines Wandaufbaus vorzunehmen.

## Features

- **Flexibler Schichtaufbau** – beliebig viele Schichten, frei benennbar
- **Temperaturverlauf** – stationäre Berechnung über Wärmedurchgangswiderstände (R = d/λ)
- **Sättigungsdampfdruck** – berechnet nach der Magnus-Formel an jeder Schichtgrenze
- **Dampfdruckverlauf** – lineare Interpolation zwischen Außen- und Innenklima
- **Taupunktprüfung** – automatische Markierung kritischer Schichtgrenzen
- **Diagramme** – Temperatur- und Dampfdruckverlauf grafisch dargestellt
- **Excel-Export** – Ergebnistabelle als `.xlsx` herunterladen

## Physikalische Grundlagen

| Größe | Formel |
|---|---|
| Wärmewiderstand | R = d / λ |
| Gesamtwiderstand | R_ges = R_si + Σ Rᵢ + R_se |
| Sättigungsdampfdruck | Magnus-Formel |
| Dampfdruck | p = φ · p_sat(T) |
| Kondensationsbedingung | p(x) > p_sat(T(x)) |

Oberflächenwiderstände nach DIN 4108-4: R_si = 0,13 m²K/W, R_se = 0,04 m²K/W.

## Installation

```bash
git clone https://github.com/dein-name/dein-repo.git
cd dein-repo
pip install -r requirements.txt
streamlit run glaser_verfahren.py
```

## requirements.txt

```
streamlit
numpy
matplotlib
pandas
astropy
openpyxl
```

## Deployment

Die App lässt sich direkt über die [Streamlit Community Cloud](https://share.streamlit.io) deployen – GitHub-Repository verbinden, `glaser_verfahren.py` als Einstiegspunkt auswählen, fertig.
