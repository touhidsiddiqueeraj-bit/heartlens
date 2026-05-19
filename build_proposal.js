const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TableOfContents, TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// Colors
const RED = "C0392B";
const TEAL = "1A6B6B";
const GOLD = "B8860B";
const DARK = "1A1410";
const LIGHT_RED = "FAE5E3";
const LIGHT_TEAL = "D0E8E8";
const LIGHT_GOLD = "FEF9ED";
const LIGHT_GRAY = "F5F5F5";
const HEADER_BG = "2C3E50";
const WHITE = "FFFFFF";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, bold: true, size: 36, color: DARK, font: "Arial" })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 28, color: RED, font: "Arial" })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 24, color: TEAL, font: "Arial" })]
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    children: [new TextRun({ text, size: 22, font: "Arial", color: "3D3530", ...opts })]
  });
}

function paraRuns(runs) {
  return new Paragraph({
    spacing: { before: 60, after: 120 },
    children: runs.map(r => new TextRun({ size: 22, font: "Arial", color: "3D3530", ...r }))
  });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 22, font: "Arial", color: "3D3530" })]
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function sectionDivider(label, number) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: noBorders,
            shading: { fill: DARK, type: ShadingType.CLEAR },
            margins: { top: 120, bottom: 120, left: 240, right: 240 },
            width: { size: 9360, type: WidthType.DXA },
            children: [new Paragraph({
              children: [
                new TextRun({ text: `${number}  `, size: 20, font: "Courier New", color: "C0392B" }),
                new TextRun({ text: label, size: 26, bold: true, font: "Arial", color: WHITE }),
              ]
            })]
          })
        ]
      })
    ]
  });
}

function calloutBox(label, text, fillColor = LIGHT_RED, borderColor = RED) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [180, 9180],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: { ...noBorders, left: { style: BorderStyle.SINGLE, size: 16, color: borderColor } },
            shading: { fill: borderColor, type: ShadingType.CLEAR },
            width: { size: 180, type: WidthType.DXA },
            children: [new Paragraph({ children: [] })]
          }),
          new TableCell({
            borders: { ...noBorders, left: { style: BorderStyle.NONE } },
            shading: { fill: fillColor, type: ShadingType.CLEAR },
            margins: { top: 100, bottom: 100, left: 180, right: 180 },
            width: { size: 9180, type: WidthType.DXA },
            children: [
              new Paragraph({ spacing: { before: 0, after: 60 }, children: [new TextRun({ text: label.toUpperCase(), size: 16, font: "Courier New", color: "7A6E68", bold: true })] }),
              new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun({ text, size: 22, font: "Arial", color: DARK })] }),
            ]
          })
        ]
      })
    ]
  });
}

function makeTable(headers, rows, colWidths, headerColor = HEADER_BG) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders,
      shading: { fill: headerColor, type: ShadingType.CLEAR },
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 20, font: "Arial", color: WHITE })] })]
    }))
  });
  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders,
      shading: { fill: ri % 2 === 0 ? "F9F9F9" : WHITE, type: ShadingType.CLEAR },
      width: { size: colWidths[ci], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      verticalAlign: VerticalAlign.TOP,
      children: Array.isArray(cell)
        ? cell
        : [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20, font: "Arial", color: "3D3530" })] })]
    }))
  }));
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

function space(n = 1) {
  return new Paragraph({ spacing: { before: 0, after: n * 120 }, children: [] });
}

// ─── DOCUMENT ────────────────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 36, bold: true, font: "Arial", color: DARK }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Arial", color: RED }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: "Arial", color: TEAL }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RED, space: 1 } },
          children: [
            new TextRun({ text: "HeartLens AI — Project Proposal", bold: true, size: 18, font: "Arial", color: DARK }),
            new TextRun({ text: "\tEdge ECG Monitoring System", size: 18, font: "Arial", color: "7A6E68" }),
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
          children: [
            new TextRun({ text: "Confidential — For Review Only", size: 16, font: "Arial", color: "7A6E68" }),
            new TextRun({ text: "\tPage ", size: 16, font: "Arial", color: "7A6E68" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: "7A6E68" }),
          ]
        })]
      })
    },
    children: [

      // ── COVER PAGE ──────────────────────────────────────────────────
      new Paragraph({ spacing: { before: 720, after: 0 }, children: [new TextRun({ text: "PROJECT PROPOSAL", size: 20, font: "Courier New", color: "7A6E68", bold: true })] }),
      new Paragraph({ spacing: { before: 120, after: 0 }, children: [new TextRun({ text: "HeartLens AI", size: 72, bold: true, font: "Arial", color: DARK })] }),
      new Paragraph({ spacing: { before: 60, after: 0 }, children: [new TextRun({ text: "Edge ECG Monitoring with On-Device AI Inference", size: 30, font: "Arial", color: RED, italics: true })] }),
      space(1),
      new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: DARK, space: 1 } }, spacing: { before: 0, after: 240 }, children: [] }),
      space(1),
      makeTable(
        ["Field", "Details"],
        [
          ["Concept", "Offline-capable ECG screening device with edge AI inference"],
          ["Target Cost", "~$12 USD (full bill of materials per unit)"],
          ["Connectivity", "None required — fully offline operation"],
          ["Device Class", "Preventative screening aid (non-diagnostic)"],
          ["MCU Platform", "ESP32 (dual-core, 240 MHz, 520 KB RAM)"],
          ["AI Models", "LSTM denoiser + 1D-CNN classifier (TFLite Micro, int8)"],
          ["Detects", "AFib, PVC, Tachycardia, Bradycardia, ST abnormality, Normal"],
          ["Build Timeline", "16 weeks (part-time)"],
          ["Open Source", "All hardware, firmware, models, and documentation"],
          ["Publication Target", "Sensors (MDPI, Q2) / IEEE IoT Journal"],
        ],
        [2800, 6560], "1A1410"
      ),
      space(2),

      // Pull quote
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [9360],
        rows: [new TableRow({ children: [new TableCell({
          borders: noBorders,
          shading: { fill: "F9F2EA", type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 360, right: 360 },
          width: { size: 9360, type: WidthType.DXA },
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: '"Every year, millions of people suffer preventable heart attacks not because treatment does not exist, but because warning signs went unnoticed until too late. HeartLens AI is built to close that gap — for anyone, anywhere, at a cost lower than a restaurant meal."', size: 24, font: "Arial", italics: true, color: DARK })]
          })]
        })] })]
      }),
      space(3),
      pageBreak(),

      // ── TABLE OF CONTENTS ────────────────────────────────────────────
      h1("Table of Contents"),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
      pageBreak(),

      // ── SECTION 1: MOTIVATION ────────────────────────────────────────
      sectionDivider("Motivation: The Problem We Are Solving", "01"),
      space(1),
      h2("Background"),
      para("Cardiovascular disease is the leading cause of death globally, responsible for approximately 18 million deaths per year. What makes this statistic particularly painful is that a large proportion of serious cardiac events — heart attacks, strokes, sudden cardiac arrest — are preceded by detectable warning signs that appear weeks or months in advance."),
      para("Conditions like atrial fibrillation (AFib), abnormal heart rhythms (arrhythmias), and ST-segment irregularities show up clearly in an electrocardiogram (ECG). Catching them early means a doctor can intervene with medication, lifestyle changes, or monitoring — before a manageable condition becomes a life-threatening emergency."),
      space(1),
      calloutBox("Core Problem", "The technology to detect these warning signs has existed for decades. The problem is access. A clinical ECG requires a hospital visit, specialist equipment, and trained staff to interpret results. For billions of people in rural areas, low-income settings, or healthcare-underserved regions, that visit simply does not happen — until something goes wrong.", LIGHT_RED, RED),
      space(1),
      h2("Gap in Existing Solutions"),
      para("Existing consumer cardiac devices (AliveCor KardiaMobile, Withings ScanWatch) cost $100–$400, require a smartphone, and depend on cloud connectivity for AI analysis. They were designed for the worried-well in wealthy urban markets."),
      para("They were not designed for a 58-year-old farmer in a village without reliable internet, or a night-shift factory worker who has never had reason to think about his heart rhythm. HeartLens AI is designed for those people."),
      para("The goal is not to replace a cardiologist. The goal is to be the thing that tells someone: 'You should go see a cardiologist' — before they end up in an ambulance."),
      space(2),
      pageBreak(),

      // ── SECTION 2: PROJECT GOALS ─────────────────────────────────────
      sectionDivider("Project Goals & Objectives", "02"),
      space(1),
      h2("Primary Goal"),
      para("HeartLens AI aims to build, validate, and document a fully offline, sub-$15 ECG monitoring device that runs AI inference directly on the hardware and outputs plain-language risk assessments — no smartphone, no internet, no medical training required."),
      space(1),
      calloutBox("Primary Objective", "Demonstrate that edge AI inference on a $3.50 microcontroller can provide meaningful cardiac screening signal — reliably distinguishing normal rhythms from patterns that warrant medical follow-up — in real-world noisy conditions.", LIGHT_TEAL, TEAL),
      space(1),
      h2("Secondary Goals"),
      bullet("Produce a rigorous noise robustness study comparing real-world vs. synthetic ECG noise conditions"),
      bullet("Open-source all hardware designs, firmware, trained models, and documentation"),
      bullet("Demonstrate deployment viability through a small real-world feasibility study"),
      bullet("Contribute a publishable research artifact to the IoT health monitoring literature"),
      bullet("Target publication in Sensors (MDPI, Q2) or IEEE Internet of Things Journal"),
      space(1),
      h2("Device Scope — What It Does and Does Not Do"),
      makeTable(
        ["The Device DOES", "The Device Does NOT"],
        [
          ["Detect rhythm patterns from 6 known categories", "Diagnose any specific medical condition"],
          ["Flag unusual patterns that may warrant a doctor visit", "Replace a cardiologist or clinical ECG"],
          ["Operate fully offline with no cloud connectivity", "Store or transmit patient data anywhere"],
          ["Output plain-language messages anyone can understand", "Show raw ECG waveforms or medical numbers to users"],
          ["Run for hours on a rechargeable LiPo battery", "Require a smartphone or internet connection"],
        ],
        [4680, 4680]
      ),
      space(2),
      pageBreak(),

      // ── SECTION 3: HOW IT WORKS ──────────────────────────────────────
      sectionDivider("System Architecture & Signal Pipeline", "03"),
      space(1),
      h2("End-to-End Signal Processing Pipeline"),
      para("The device captures an electrical signal from the skin surface (the same signal measured in hospital ECGs), cleans it using an AI denoiser running locally on the chip, classifies the rhythm pattern against known cardiac conditions, and displays a plain-language result on a small screen. The entire process happens in under 100 milliseconds, with no data ever leaving the device."),
      space(1),
      makeTable(
        ["Stage", "Component", "Function", "Latency"],
        [
          ["1 — Capture", "Skin electrodes + AD8232", "Amplifies mV-level skin electrical signal; analog band-pass filters baseline drift and 50/60 Hz interference before ADC", "Hardware"],
          ["2 — Sampling", "ESP32 ADC (12-bit)", "Samples at 360 Hz; stores rolling 10-second window in SRAM as int16 array", "~0 ms"],
          ["3 — Denoising", "LSTM denoiser (TFLite Micro)", "Sequence-to-sequence network removes motion artifacts, EMI, muscle noise from each window", "~30 ms"],
          ["4 — Classification", "1D-CNN classifier (TFLite Micro)", "Identifies rhythm from 6 categories; outputs class + confidence score", "~20 ms"],
          ["5 — Interpretation", "Rule-based C++ interpreter", "Maps class + confidence to one of 4 plain-language outputs; flags uncertain results", "< 1 ms"],
          ["6 — Display", "SSD1306 OLED (I2C)", "Renders plain-language message; shows battery level indicator", "< 5 ms"],
        ],
        [1400, 2200, 4360, 1200]
      ),
      space(1),
      h2("Rhythm Classification Categories"),
      para("The classifier recognizes six rhythm categories, each mapped to a plain-language output and urgency tier:"),
      space(1),
      makeTable(
        ["Rhythm Category", "Description", "Urgency", "User Message"],
        [
          ["Normal Sinus Rhythm", "Regular rate, normal P-QRS-T morphology", "None", "Heart rhythm looks normal."],
          ["Atrial Fibrillation (AFib)", "Irregular rhythm, absent P waves, variable R-R intervals", "High", "Irregular rhythm detected. Please seek medical attention."],
          ["Premature Ventricular Contraction", "Early wide QRS complex, compensatory pause", "Medium", "Unusual rhythm detected. Please see a doctor soon."],
          ["Tachycardia", "Heart rate > 100 bpm with consistent morphology", "Medium", "Unusual rhythm detected. Please see a doctor soon."],
          ["Bradycardia", "Heart rate < 50 bpm with consistent morphology", "Medium", "Unusual rhythm detected. Please see a doctor soon."],
          ["ST Abnormality", "Elevated or depressed ST segment; ischemia marker", "High", "Irregular rhythm detected. Please seek medical attention."],
        ],
        [2200, 2800, 1200, 3160]
      ),
      space(1),
      calloutBox("Plain-Language Output Design", "The interpreter never shows raw ECG data, confidence percentages, or medical terminology to the end user. The four possible outputs are: (1) Heart rhythm looks normal. (2) Unusual rhythm detected. Please see a doctor soon. (3) Irregular rhythm detected. Please seek medical attention. (4) Signal unclear. Please reattach electrodes and try again.", LIGHT_GOLD, GOLD),
      space(2),
      pageBreak(),

      // ── SECTION 4: HARDWARE STACK ────────────────────────────────────
      sectionDivider("Hardware Stack & Bill of Materials", "04"),
      space(1),
      h2("Component Overview"),
      para("Every component was selected to minimize cost while maintaining clinical-grade signal quality sufficient for rhythm classification. The total bill of materials (BOM) for a single unit targets under $15 USD, sourced from standard electronics suppliers (AliExpress, LCSC, Digi-Key)."),
      space(1),
      makeTable(
        ["Component", "Model / Spec", "Role", "Unit Cost (USD)"],
        [
          ["Microcontroller", "ESP32-WROOM-32 (or Dev board)", "240 MHz dual-core, 520 KB RAM, 4 MB Flash. Runs all AI inference and firmware. Built-in BT/Wi-Fi (unused).", "$3.50"],
          ["ECG Analog Frontend", "AD8232 (SparkFun breakout)", "Instrumentation amplifier + band-pass filter. Amplifies mV-level cardiac signals. Built-in right-leg-drive circuit.", "$2.00"],
          ["OLED Display", "SSD1306, 0.96\", 128x64 px, I2C", "Low-power monochrome display. Shows plain-language result. Visible in sunlight. No backlight drain.", "$1.50"],
          ["Electrodes & Cable", "Snap-type reusable cable + disposable gel pads", "Adhesive patches placed on chest or wrists. Snap connector for reusable lead wire. Gel pads ~$0.10 each.", "$1.50"],
          ["Power — Battery", "LiPo 500–1000 mAh (3.7V)", "Portable rechargeable power. 500 mAh gives ~6 hours continuous use at 80 mA average draw.", "$1.50"],
          ["Power — Charger", "TP4056 USB-C charge module", "LiPo charge controller with overcharge and over-discharge protection. Standard micro-USB or USB-C.", "$0.50"],
          ["Passives", "Resistors, capacitors, connectors", "Decoupling caps on AD8232 supply rails; RC filter on ADC input; headers and JST connectors.", "$0.50"],
          ["PCB", "Custom 2-layer, 60x40 mm (JLCPCB)", "Eliminates breadboard unreliability. $2 for 5 boards from JLCPCB at minimum order.", "$0.40"],
          ["", "", "TOTAL BOM (single unit)", "~$11.90"],
        ],
        [2000, 2200, 3700, 1460]
      ),
      space(1),
      h2("Circuit Description"),
      h3("Analog Signal Path"),
      para("The AD8232 is a purpose-built integrated circuit for ECG signal conditioning. It contains a fully differential instrumentation amplifier with configurable gain, an operational amplifier for additional gain stages, a right-leg-drive (RLD) circuit to reduce common-mode interference, and an internal band-pass filter."),
      para("The signal path from skin to ADC is as follows: (1) Differential voltage from two electrodes enters the AD8232 IN+ and IN- pins. (2) The RLD electrode on the third lead (right leg or right arm) actively drives the common-mode voltage toward zero. (3) The AD8232 amplifies the differential signal by approximately 100x and band-pass filters to 0.5–40 Hz (the clinically relevant ECG frequency range). (4) The OUTPUT pin connects to GPIO34 of the ESP32 (ADC1 channel). (5) The LOFF+ and LOFF- pins monitor electrode contact; firmware reads these to detect disconnected leads."),
      space(1),
      h3("Digital Processing Path"),
      para("The ESP32 ADC samples at 360 Hz (matching the MIT-BIH Arrhythmia Database native sample rate). A rolling circular buffer stores the most recent 1296 samples (3.6 seconds — approximately 3–5 heartbeats). The firmware runs a 10-second sliding window analysis loop, feeding 3600-sample windows into the TFLite Micro inference engine."),
      space(1),
      h3("Power System"),
      para("The TP4056 charge controller accepts USB 5V input and manages LiPo charging at up to 1A. A low-dropout 3.3V regulator (AMS1117-3.3) supplies the ESP32 and AD8232. A voltage divider on GPIO35 monitors the battery voltage. The firmware displays low-battery warning and disables inference at < 3.4V to prevent ADC noise from LiPo voltage sag corrupting readings."),
      space(2),
      pageBreak(),

      // ── SECTION 5: SOFTWARE STACK ────────────────────────────────────
      sectionDivider("Software Stack & AI Architecture", "05"),
      space(1),
      h2("Two-Environment Architecture"),
      para("The software exists in two separate environments: a training environment (Python on desktop/cloud GPU, used once) and a deployment environment (C++ on ESP32, runs permanently)."),
      space(1),
      makeTable(
        ["Layer", "Environment", "Technology", "Purpose"],
        [
          ["Data Ingestion", "Training (Python)", "WFDB library, NumPy, SciPy", "Parse MIT-BIH Arrhythmia Database (48 annotated 30-min recordings). Extract beat-labeled segments at 360 Hz. Apply augmentation and noise injection pipeline."],
          ["Noise Injection", "Training (Python)", "NumPy, custom pipeline", "Add motion artifact, baseline wander, 50/60 Hz PLI, EMG at SNR levels from 0 dB to 40 dB. Creates 5x data augmentation for denoiser training."],
          ["Denoiser Model", "Training + Deployment", "Keras LSTM → TFLite Micro (int8)", "Seq-to-seq LSTM: 2 layers, 64 hidden units, input window 360 samples. Trained to reconstruct clean ECG from noisy input. Post-training int8 quantization to 148 KB."],
          ["Classifier Model", "Training + Deployment", "Keras 1D-CNN → TFLite Micro (int8)", "1D-CNN: 3 conv blocks (32/64/128 filters, kernel 5), global average pool, softmax head (6 classes). Trained on denoised segments. Quantized to 112 KB."],
          ["Quantization", "Optimization (Python)", "TensorFlow Lite post-training int8", "Reduces model size ~4x and inference time ~3-4x. Full integer quantization with representative dataset calibration. Both models fit in ESP32 SRAM simultaneously."],
          ["Firmware", "Deployment (C++)", "Arduino framework + ESP-IDF", "State machine: IDLE → SAMPLING → INFERENCE → DISPLAY → IDLE. FreeRTOS tasks: ADC sampling (Core 0), inference + display (Core 1). Deterministic real-time loop."],
          ["Interpreter", "Deployment (C++)", "Rule-based logic", "Confidence threshold: > 0.75 triggers result display; 0.55–0.75 triggers 'signal unclear'; < 0.55 requests electrode reattachment. Maps class to urgency tier and message string."],
          ["Model Tools", "Training (Python)", "TensorFlow, Keras, sklearn", "Training loop, early stopping, confusion matrix evaluation, ROC-AUC per class, McNemar test for statistical comparison vs. baseline."],
        ],
        [1600, 1600, 2400, 3760]
      ),
      space(1),
      h2("Model Architecture Details"),
      h3("LSTM Denoiser"),
      para("Architecture: Encoder-decoder LSTM. Input: 360-sample window (1 second at 360 Hz), normalized to [-1, 1]. Encoder: 2 LSTM layers (64 units each, return_sequences=True). Decoder: 2 LSTM layers (64 units) + Dense output layer (360 units, linear activation). Loss: MSE on clean ECG reconstruction. Training data: MIT-BIH recordings + synthetic noise at 8 SNR levels. Post-quantization size: ~148 KB."),
      space(1),
      h3("1D-CNN Classifier"),
      para("Architecture: Three convolutional blocks. Each block: Conv1D → BatchNorm → ReLU → MaxPool. Filter counts: 32 / 64 / 128. Kernel size: 5 across all layers. Global Average Pooling after final conv block. Dense(64, ReLU) → Dropout(0.5) → Dense(6, Softmax). Input: denoised 360-sample window. Output: 6-class probability vector. Post-quantization size: ~112 KB."),
      space(1),
      calloutBox("Memory Budget", "ESP32 total SRAM: 520 KB. Allocation: Denoiser model: 148 KB. Classifier model: 112 KB. Sample buffer (3600 x int16): ~7 KB. FreeRTOS overhead: ~40 KB. Display buffer: ~1 KB. Available headroom: ~212 KB. Both models fit simultaneously with no flash overlay needed.", LIGHT_TEAL, TEAL),
      space(2),
      pageBreak(),

      // ── SECTION 6: CIRCUIT DIAGRAM (ASCII/TEXT) ──────────────────────
      sectionDivider("Circuit Schematic Overview", "06"),
      space(1),
      h2("System Block Diagram"),
      para("The following table represents the system interconnection schematic. A full KiCad schematic (.sch) and PCB layout (.kicad_pcb) will be produced in Sprint 1 and open-sourced with the project."),
      space(1),
      makeTable(
        ["Block", "Signals / Connections", "Notes"],
        [
          ["Electrode LA (Left Arm)", "AD8232 IN+", "Positive differential ECG input"],
          ["Electrode RA (Right Arm)", "AD8232 IN-", "Negative differential ECG input"],
          ["Electrode RL (Right Leg)", "AD8232 SDN/RLD", "Right-leg drive — common-mode rejection"],
          ["AD8232 OUTPUT", "ESP32 GPIO34 (ADC1_CH6)", "Filtered, amplified ECG signal ~0.5–2.5 V"],
          ["AD8232 LOFF+", "ESP32 GPIO32", "Lead-off detection positive"],
          ["AD8232 LOFF-", "ESP32 GPIO33", "Lead-off detection negative"],
          ["AD8232 VCC", "3.3 V rail", "100 nF decoupling cap to GND at pin"],
          ["SSD1306 SDA", "ESP32 GPIO21 (I2C SDA)", "Display data line — 4.7 kOhm pull-up to 3.3 V"],
          ["SSD1306 SCL", "ESP32 GPIO22 (I2C SCL)", "Display clock line — 4.7 kOhm pull-up to 3.3 V"],
          ["SSD1306 VCC", "3.3 V rail", "100 nF decoupling cap"],
          ["TP4056 VOUT+", "LiPo BAT+ → AMS1117 VIN", "4.2 V max LiPo output"],
          ["AMS1117-3.3 VOUT", "ESP32 3V3, AD8232 VCC, SSD1306 VCC", "Regulated 3.3 V supply to all digital components"],
          ["LiPo BAT+ (via divider)", "ESP32 GPIO35 (ADC1_CH7)", "10k/10k divider — battery voltage monitor"],
          ["USB-C connector VBUS", "TP4056 VIN", "5 V USB charging input"],
          ["ESP32 EN button", "ESP32 EN pin → 10k pull-up to 3.3 V", "Hardware reset button"],
        ],
        [2400, 3000, 3960]
      ),
      space(1),
      calloutBox("PCB Notes", "Target PCB dimensions: 60 x 40 mm (fits inside a standard project enclosure). 2-layer stackup. Signal traces on Top copper; power and ground pour on Bottom copper. Electrode connector: 3-pin 2.54 mm JST-PH. Battery connector: 2-pin 2.0 mm JST-PH. Display connector: 4-pin 2.54 mm header. Manufacturing: JLCPCB (5 boards, ~$10 total including shipping).", LIGHT_GOLD, GOLD),
      space(2),
      pageBreak(),

      // ── SECTION 7: COST ANALYSIS ─────────────────────────────────────
      sectionDivider("Cost Analysis & Budget", "07"),
      space(1),
      h2("Bill of Materials — Detailed Cost"),
      makeTable(
        ["Item", "Quantity", "Unit Cost (USD)", "Total (USD)", "Supplier"],
        [
          ["ESP32-WROOM-32D module", "3", "$2.80", "$8.40", "LCSC / AliExpress"],
          ["AD8232 ECG breakout board", "3", "$1.80", "$5.40", "AliExpress"],
          ["SSD1306 OLED 0.96\" I2C", "3", "$1.20", "$3.60", "AliExpress"],
          ["LiPo battery 800 mAh 3.7V", "3", "$1.50", "$4.50", "AliExpress"],
          ["TP4056 USB-C charge module", "3", "$0.40", "$1.20", "AliExpress"],
          ["AMS1117-3.3 SOT-223", "10", "$0.10", "$1.00", "LCSC"],
          ["Electrode snap cable (3-lead)", "3", "$1.00", "$3.00", "AliExpress"],
          ["Disposable gel electrode pads", "50", "$0.10", "$5.00", "Amazon"],
          ["Resistors assortment (SMD 0402)", "1 kit", "–", "$3.00", "LCSC"],
          ["Capacitors assortment (SMD 0402)", "1 kit", "–", "$3.00", "LCSC"],
          ["JST connectors (PH + ZH)", "10 sets", "–", "$2.00", "AliExpress"],
          ["Custom PCB (JLCPCB, 5 boards)", "1 order", "–", "$9.50", "JLCPCB"],
          ["Dev board headers + misc.", "–", "–", "$3.00", "Local"],
          ["", "", "SUBTOTAL (3 prototypes)", "$53.60", ""],
          ["", "", "Contingency (+15%)", "$8.04", ""],
          ["", "", "TOTAL BUDGET", "$61.64", ""],
          ["", "", "Cost per prototype unit", "~$20.55", "(includes shared tooling)"],
        ],
        [2600, 800, 1600, 1600, 2760]
      ),
      space(1),
      h2("Development & Non-Hardware Costs"),
      makeTable(
        ["Item", "Cost", "Notes"],
        [
          ["GPU compute — model training", "$0", "Google Colab free tier (T4 GPU, estimated 4–8 hours total)"],
          ["MIT-BIH Arrhythmia Database", "$0", "Freely available on PhysioNet (open access)"],
          ["Software toolchain", "$0", "All open-source: TensorFlow, Arduino IDE, KiCad, Python"],
          ["MDPI Sensors publication APC", "~$2,200", "Article Processing Charge if accepted (fee waiver may apply for student authors)"],
          ["IRB / ethics review (feasibility study)", "~$0–$50", "University ethics board — typically waived for low-risk observational studies"],
          ["Miscellaneous (shipping, tools)", "~$30", "Multimeter, oscilloscope probe, soldering consumables"],
        ],
        [3200, 1600, 4560]
      ),
      space(1),
      h2("Cost Comparison — Market Context"),
      makeTable(
        ["Product", "Price", "Connectivity Required", "Offline AI", "Target User"],
        [
          ["HeartLens AI (this project)", "~$12 BOM", "None", "Yes (on-device)", "Underserved / rural / no smartphone"],
          ["AliveCor KardiaMobile", "$99 + $99/yr sub", "Smartphone required", "No (cloud)", "Smartphone users, urban"],
          ["Withings ScanWatch", "$299", "Smartphone required", "No (cloud)", "Affluent tech consumers"],
          ["Clinical ECG (hospital)", "$50–$200/session", "Hospital visit", "N/A", "Patients with insurance"],
          ["Holter Monitor (24-hr rental)", "$200–$500", "Hospital infrastructure", "N/A", "High-risk cardiac patients"],
        ],
        [2400, 1400, 1600, 1200, 2760]
      ),
      space(2),
      pageBreak(),

      // ── SECTION 8: GANTT CHART ───────────────────────────────────────
      sectionDivider("Project Timeline — 16-Week Gantt Chart", "08"),
      space(1),
      h2("Sprint Overview"),
      para("The project is organized into 5 sprints across 16 weeks. Each sprint has a defined deliverable milestone. Work is part-time (~10–15 hours per week alongside university coursework)."),
      space(1),
      makeTable(
        ["Sprint", "Weeks", "Theme", "Key Deliverable"],
        [
          ["Sprint 1", "W1–W3", "Hardware & Signal Acquisition", "Prototype board assembled; clean ECG signal verified on oscilloscope"],
          ["Sprint 2", "W4–W7", "Data Pipeline & Model Training", "Both AI models trained, quantized, and benchmarked on MIT-BIH"],
          ["Sprint 3", "W8–W10", "Firmware Integration", "Full pipeline running on ESP32; latency and RAM measurements recorded"],
          ["Sprint 4", "W11–W13", "Noise Study & Feasibility Testing", "Real-world noise study complete; small participant study conducted"],
          ["Sprint 5", "W14–W16", "Documentation & Publication", "Paper draft submitted; all hardware/code open-sourced on GitHub"],
        ],
        [1000, 1000, 2400, 4960]
      ),
      space(1),
      h2("Detailed Week-by-Week Gantt"),
      makeTable(
        ["Task", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "W13", "W14", "W15", "W16"],
        [
          // Hardware
          ["KiCad Schematic + PCB layout", "██", "██", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
          ["PCB order + component sourcing", "", "██", "██", "", "", "", "", "", "", "", "", "", "", "", "", ""],
          ["PCB assembly + AD8232 bringup", "", "", "██", "", "", "", "", "", "", "", "", "", "", "", "", ""],
          ["Signal quality verification (scope)", "", "", "██", "", "", "", "", "", "", "", "", "", "", "", "", ""],
          // Data
          ["MIT-BIH data download + WFDB parse", "", "", "", "██", "", "", "", "", "", "", "", "", "", "", "", ""],
          ["Noise injection pipeline", "", "", "", "██", "██", "", "", "", "", "", "", "", "", "", "", ""],
          ["Denoiser LSTM training", "", "", "", "", "██", "██", "", "", "", "", "", "", "", "", "", ""],
          ["CNN classifier training", "", "", "", "", "", "██", "██", "", "", "", "", "", "", "", "", ""],
          ["int8 quantization + size check", "", "", "", "", "", "", "██", "", "", "", "", "", "", "", "", ""],
          // Firmware
          ["ADC sampling firmware (FreeRTOS)", "", "", "", "", "", "", "", "██", "", "", "", "", "", "", "", ""],
          ["TFLite Micro integration", "", "", "", "", "", "", "", "██", "██", "", "", "", "", "", "", ""],
          ["Interpreter + display firmware", "", "", "", "", "", "", "", "", "██", "", "", "", "", "", "", ""],
          ["Full pipeline integration + latency test", "", "", "", "", "", "", "", "", "██", "██", "", "", "", "", "", ""],
          // Testing
          ["Real-world noise capture study", "", "", "", "", "", "", "", "", "", "██", "██", "", "", "", "", ""],
          ["Participant feasibility study (n=10)", "", "", "", "", "", "", "", "", "", "", "██", "██", "", "", "", ""],
          ["Confusion matrix + per-class analysis", "", "", "", "", "", "", "", "", "", "", "", "██", "", "", "", ""],
          // Publication
          ["Paper draft (methods + results)", "", "", "", "", "", "", "", "", "", "", "", "", "██", "██", "", ""],
          ["Open-source release (GitHub)", "", "", "", "", "", "", "", "", "", "", "", "", "", "██", "██", ""],
          ["Paper submission + revision", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "██", "██"],
        ],
        [2500, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395, 395]
      ),
      space(1),
      calloutBox("Milestone Gates", "Sprint 1 exit gate: Clean ECG visible on oscilloscope (SNR > 20 dB, baseline stable). Sprint 2 exit gate: Classifier achieves F1 > 0.87 on clean MIT-BIH test set. Sprint 3 exit gate: Full pipeline latency < 100 ms on hardware. Sprint 4 exit gate: Accuracy on real-world capture within 8% of clean benchmark. Sprint 5 exit gate: Paper submitted to Sensors or IEEE IoT Journal.", LIGHT_RED, RED),
      space(2),
      pageBreak(),

      // ── SECTION 9: RESEARCH CONTRIBUTION ────────────────────────────
      sectionDivider("Research Contribution & Publication Strategy", "09"),
      space(1),
      h2("Three Distinct Contributions"),
      h3("1. Full-Stack Edge Inference Within Hard Hardware Constraints"),
      para("Prior work typically benchmarks models on desktop hardware and claims 'deployability.' This project implements and measures the complete pipeline — denoise, classify, interpret — running simultaneously on a $3.50 microcontroller, with real latency and RAM measurements published."),
      h3("2. Systematic Real-World Noise Robustness Study"),
      para("The noise injection pipeline (motion artifact, baseline wander, 50/60 Hz interference, muscle noise at varying SNR levels) is compared against ECG captured from the actual prototype in real conditions. This synthetic vs. real-world noise comparison is underrepresented in embedded ECG literature."),
      h3("3. Deployment-Framed Feasibility Study"),
      para("A small study with real participants produces honest end-to-end accuracy benchmarks — not 'accuracy on a clean benchmark dataset' but 'accuracy when a real person attaches electrodes in a non-clinical setting.' This is the number that actually matters for real-world deployment."),
      space(1),
      makeTable(
        ["Target Venue", "Type", "Scope Fit", "Impact Factor"],
        [
          ["Sensors (MDPI)", "Primary — Journal", "IoT + biomedical sensing, edge AI, open hardware", "Q2, IF ~3.9"],
          ["IEEE Internet of Things Journal", "Alternate — Journal", "Edge computing, embedded systems, healthcare IoT", "Q1, IF ~10.6"],
          ["Biomedical Signal Processing & Control", "Alternate — Journal", "ECG signal processing, clinical biomedical engineering", "Q1, IF ~5.1"],
          ["IEEE EMBC Conference", "Conference (backup)", "Biomedical engineering — poster or short paper track", "Top-tier in field"],
        ],
        [3000, 1400, 3000, 1960]
      ),
      space(1),
      calloutBox("Regulatory Scope — Explicitly Stated in the Paper", "HeartLens AI is a preventative screening aid, not a medical device. It is not intended to diagnose, treat, or manage any condition. Its sole function is to flag rhythm patterns that may warrant professional evaluation. All performance claims will be scoped accordingly. This framing is not a limitation — it is the correct and honest characterization of what the device does, and reviewers in this space respond well to authors who demonstrate they understand the regulatory distinction.", LIGHT_TEAL, TEAL),
      space(2),
      pageBreak(),

      // ── SECTION 10: RISKS ────────────────────────────────────────────
      sectionDivider("Risk Register & Mitigation", "10"),
      space(1),
      makeTable(
        ["Risk", "Likelihood", "Impact", "Mitigation"],
        [
          ["SRAM overflow: both models exceed 520 KB", "Medium", "High", "Pre-calculated: 260 KB total + OS overhead fits. Fallback: use flash overlay (esp_partition_mmap) to page denoiser in/out. Last resort: switch to lighter MobileNet-style architecture."],
          ["AD8232 signal quality insufficient in field", "Medium", "High", "Test on oscilloscope at Sprint 1 gate. If SNR < 15 dB, add external instrumentation amp stage (INA333). Spiral electrode pattern as backup for wrist capture."],
          ["MIT-BIH class imbalance (AFib << Normal)", "High", "Medium", "Apply SMOTE oversampling and class-weighted loss during training. Validate per-class F1, not just accuracy. Target F1 > 0.85 on minority classes."],
          ["Participant recruitment for feasibility study", "Low–Medium", "Medium", "University ethics board approval first. Target 10 healthy volunteers (colleagues, family). Expand to 20 if time allows. Study is observational — minimal IRB complexity."],
          ["Quantization accuracy degradation > 5%", "Medium", "Medium", "Use representative calibration dataset (500 samples per class). If degradation > 5% on any class, apply quantization-aware training (QAT) in Keras before re-exporting."],
          ["PCB manufacturing defect / component failure", "Low", "Medium", "Order 5 PCBs (JLCPCB minimum order), 3 of each component. Breadboard fallback available for all Sprint 2 work. Keep 2 spare ESP32 boards."],
          ["Scope creep delaying publication", "High", "Medium", "Strict sprint gates. Scope is fixed: no new rhythm classes, no BLE feature, no app development. Gantt reviewed weekly."],
        ],
        [2400, 1000, 900, 5060]
      ),
      space(2),
      pageBreak(),

      // ── CONCLUSION ───────────────────────────────────────────────────
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [9360],
        rows: [new TableRow({ children: [new TableCell({
          borders: noBorders,
          shading: { fill: DARK, type: ShadingType.CLEAR },
          margins: { top: 360, bottom: 360, left: 480, right: 480 },
          width: { size: 9360, type: WidthType.DXA },
          children: [
            new Paragraph({ spacing: { before: 0, after: 60 }, children: [new TextRun({ text: "07 — Conclusion", size: 20, font: "Courier New", color: RED })] }),
            new Paragraph({ spacing: { before: 0, after: 180 }, children: [new TextRun({ text: "Why This Project Matters", size: 36, bold: true, font: "Arial", color: WHITE })] }),
            new Paragraph({ spacing: { before: 0, after: 180 }, children: [new TextRun({ text: "The technology to build HeartLens AI exists today, is inexpensive, is well-understood, and is sitting unused in the gap between clinical-grade devices and the billions of people who will never have regular access to them.", size: 22, font: "Arial", color: "D0CCC8" })] }),
            new Paragraph({ spacing: { before: 0, after: 180 }, children: [new TextRun({ text: "This project does not require a breakthrough. It requires careful engineering, honest evaluation, and a willingness to build for the actual user — not the user who already has a smartphone, a cardiologist, and a stable internet connection.", size: 22, font: "Arial", color: "D0CCC8" })] }),
            new Paragraph({ spacing: { before: 0, after: 180 }, children: [new TextRun({ text: "The hardware costs ~$12. The software runs entirely offline. The output is four sentences that anyone can understand.", size: 22, font: "Arial", color: WHITE, bold: true })] }),
            new Paragraph({
              spacing: { before: 120, after: 0 },
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: "555555", space: 1 } },
              children: [new TextRun({ text: '"The gap between detectable and detected is not a scientific problem. It is an access problem. HeartLens AI is one small attempt to close it."', size: 24, font: "Arial", italics: true, color: WHITE })]
            }),
          ]
        })] })]
      }),
      space(3),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("./HeartLens_AI_Project_Proposal.docx", buffer);
  console.log("Done.");
});
