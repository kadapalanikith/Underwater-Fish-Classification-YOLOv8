import React from 'react';
import { motion } from 'framer-motion';
import { FileText, BookOpen, Download } from 'lucide-react';

const SAMPLE_PDF = '/research/sample_paper.pdf';

const PAPERS = [
  {
    title: 'Project validation status',
    journal: 'Research report / publication: not verified',
    year: '2026',
    abstract: 'The saved YOLOv8 checkpoint was re-evaluated on the local test split. test_full (697 images, 1,129 instances) reached mAP@50 0.840 and mAP@50–95 0.622; the 399-image leakage-filtered test_clean subset (639 instances) reached 0.841 and 0.636. Image-level 95% bootstrap intervals for test_clean were 0.823–0.880 and 0.615–0.694. The original notebook validation log remains 0.865/0.874/0.908/0.666. The test split shares exact duplicates and source stems with training data, so test_full is not fully independent; test_clean is a conservative subset. A separate zero-shot OBSEA check reached class-agnostic mAP@50 0.036 and mAP@50–95 0.017. Some classes have very low support, and the repository does not establish publication or peer review.',
    status: 'project audit',
  },
  {
    title: 'RT-DETR comparison',
    journal: 'Planned; no fine-tuning run executed',
    year: '2026',
    abstract: 'The saved RT-DETR-L checkpoint is COCO-pretrained and has 80 classes, not fish classes. The recorded comparison is timing-only. A matched fish fine-tuning and evaluation plan is documented in docs/rtdetr-finetune-plan.md, but no compatible GPU environment, training run, or fish accuracy result exists yet.',
    status: 'not executed',
  }
];

const METRICS = [
  { label: 'Held-out test mAP@50',    val: '0.841', note: '399 leakage-filtered images; 95% CI 0.823–0.880' },
  { label: 'Held-out test mAP@50–95', val: '0.636', note: 'Same 639-instance subset; 95% CI 0.615–0.694' },
  { label: 'OBSEA zero-shot mAP@50',  val: '0.036', note: '618 external images; class-agnostic; 5,369 instances' },
  { label: 'Historical validation mAP@50', val: '0.908', note: 'Original notebook log; 12/13 classes' },
];

const METHODOLOGY = [
  { step: '1', title: 'Dataset', body: 'The local dataset has 8,229 images (6,834 train / 698 valid / 697 test), all 640×640 RGB, with 13 classes. Labels mix boxes and polygons; polygons were converted to enclosing boxes for evaluation. The test split has 34 exact image-hash matches with train (33 distinct hashes) and 212 images sharing a train source stem; 70 images share a valid source stem. The leakage filter removed 298 images. The unlabeled qualitative sample images are not an independent test set.' },
  { step: '2', title: 'Training',  body: 'The original notebook trains YOLOv8n for 10 epochs at 640 px. Detailed reproducibility metadata and a newly executed training run are not available in this audit.' },
  { step: '3', title: 'Evaluation', body: 'The saved checkpoint was re-evaluated with Ultralytics on CPU (conf 0.001, IoU 0.6). test_full: mAP@50 0.840 / mAP@50–95 0.622. test_clean: 0.841 / 0.636. valid_full: 0.910 / 0.658; AngelFish has no validation instances. Image-level 95% bootstrap intervals use 2,000 resamples and require every evaluated class to be present; source-stem cluster intervals are a sensitivity check. They do not fix leakage or prove source independence.' },
  { step: '4', title: 'Application', body: 'A FastAPI backend and React frontend are present. CPU inference is verified; GPU support depends on a compatible installed environment. RT-DETR and real-video tracking have not been validated as fish-accurate.' },
];

const fadeIn = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.55 } } };

export default function ResearchPage() {
  return (
    <div style={{ paddingTop: '70px' }}>

      {/* Header */}
      <section style={{ padding: '4rem 0 3rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge">Project Audit</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3rem)', marginBottom: '1rem' }}>Research Validation Status</h1>
          <p style={{ color: '#6b7f9e', maxWidth: '540px', margin: '0 auto', fontSize: '1.05rem' }}>
            Evidence and limitations for the saved baseline and preliminary experiments. No publication is verified.
          </p>
        </div>
      </section>

          {/* Validation record */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
          <div className="badge">Validation record</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem' }}>Research status</h2>
            {PAPERS.map((p, i) => (
              <div key={i} className="card" style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-start' }}>
                  <div style={{ width: 50, height: 50, flexShrink: 0, borderRadius: 14, background: 'rgba(63,114,175,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#3F72AF' }}>
                    <FileText size={24} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem', alignItems: 'center' }}>
                      <span style={{ padding: '0.2rem 0.65rem', background: 'rgba(234,179,8,0.12)', color: '#a16207', borderRadius: 9999, fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase' }}>
                        {p.status}
                      </span>
                      <span style={{ fontSize: '0.82rem', color: '#9aafcc' }}>{p.journal}</span>
                    </div>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#112D4E', marginBottom: '0.75rem', lineHeight: 1.4 }}>{p.title}</h3>
                    <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7, marginBottom: '1rem' }}>{p.abstract}</p>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Embedded PDF Viewer */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container">
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Preserved Artifact</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '0.75rem' }}>Project Document Preview</h2>
            <p style={{ color: '#6b7f9e', fontSize: '0.9rem', lineHeight: 1.7, marginBottom: '1.5rem', maxWidth: '680px' }}>
              This PDF is preserved from the project workspace. Its contents and publication status
              have not been verified here, and it is not evidence of peer review.
            </p>
            <div className="pdf-viewer" style={{ boxShadow: '0 8px 40px rgba(17,45,78,0.1)' }}>
              <object
                data={SAMPLE_PDF}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ borderRadius: 18, display: 'block' }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem', background: '#f0f4f8', borderRadius: 18 }}>
                  <BookOpen size={64} color="#9aafcc" />
                  <p style={{ color: '#6b7f9e', fontWeight: 600 }}>PDF preview not available in this browser.</p>
                  <a href={SAMPLE_PDF} download className="btn btn-primary" style={{ padding: '0.65rem 1.5rem' }}>
                    <Download size={16} /> Download Preserved PDF
                  </a>
                </div>
              </object>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Metrics */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Evaluation</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem' }}>Performance Metrics</h2>
            <div className="stat-grid">
              {METRICS.map(m => (
                <div key={m.label} className="stat-card" style={{ borderTop: '4px solid #3F72AF' }}>
                  <div className="stat-val">{m.val}</div>
                  <div className="stat-key" style={{ marginBottom: '0.25rem' }}>{m.label}</div>
                  <div style={{ fontSize: '0.72rem', color: '#c0cfe0' }}>{m.note}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Preprocessing accuracy */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Preprocessing</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '0.75rem' }}>Enhancement Accuracy Check</h2>
            <p style={{ color: '#6b7f9e', fontSize: '0.9rem', lineHeight: 1.7, marginBottom: '1.5rem', maxWidth: '720px' }}>
              Same checkpoint and 399 leakage-filtered test images (639 instances). CLAHE's
              mAP@50 change was +0.005 with a paired 95% interval of -0.003 to +0.011; it was
              not supported after Holm correction. Gray-world and fusion reduced mAP@50–95 by
              about 0.023–0.025; those degradations remained supported after Holm correction.
              Some classes have low support. Enhancement latency was 10.5–15.1 ms per 640×640
              image on CPU.
            </p>
            <div className="stat-grid">
              {[
                { label: 'Raw (JPEG control)', val: '0.841', note: 'mAP@50 / 0.636 mAP@50–95' },
                { label: 'CLAHE', val: '0.846', note: 'mAP@50 / 0.634 mAP@50–95; not supported' },
                { label: 'Gray-world', val: '0.812', note: 'mAP@50 / 0.613 mAP@50–95; mAP50-95 drop supported' },
                { label: 'Fusion', val: '0.815', note: 'mAP@50 / 0.611 mAP@50–95; mAP50-95 drop supported' },
              ].map(m => (
                <div key={m.label} className="stat-card" style={{ borderTop: '4px solid #3F72AF' }}>
                  <div className="stat-val">{m.val}</div>
                  <div className="stat-key" style={{ marginBottom: '0.25rem' }}>{m.label}</div>
                  <div style={{ fontSize: '0.72rem', color: '#c0cfe0' }}>{m.note}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Cross-domain check */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Cross-domain check</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '0.75rem' }}>OBSEA Zero-Shot Evaluation</h2>
            <p style={{ color: '#6b7f9e', fontSize: '0.9rem', lineHeight: 1.7, marginBottom: '1.5rem', maxWidth: '760px' }}>
              The saved fish checkpoint was evaluated once on the external OBSEA Mediterranean
              observatory test split (CC BY 4.0; DOI 10.5281/zenodo.14888440). All 21 fish taxa
              were collapsed to one class, Diver and Octopus annotations were excluded, and no
              OBSEA tuning or model selection was performed. The 618-image test set contains
              5,369 fish instances after removing 4 duplicate boxes. This measures class-agnostic
              detection under domain shift, not 13-class species accuracy.
            </p>
            <div className="stat-grid">
              {[
                { label: 'OBSEA mAP@50', val: '0.036', note: 'class-agnostic; zero-shot' },
                { label: 'OBSEA mAP@50–95', val: '0.017', note: 'class-agnostic; zero-shot' },
                { label: 'Precision', val: '0.124', note: 'every prediction treated as fish' },
                { label: 'Recall', val: '0.094', note: 'P and R are not species metrics' },
              ].map(m => (
                <div key={m.label} className="stat-card" style={{ borderTop: '4px solid #a16207' }}>
                  <div className="stat-val">{m.val}</div>
                  <div className="stat-key" style={{ marginBottom: '0.25rem' }}>{m.label}</div>
                  <div style={{ fontSize: '0.72rem', color: '#c0cfe0' }}>{m.note}</div>
                </div>
              ))}
            </div>
            <p style={{ color: '#6b7f9e', fontSize: '0.78rem', lineHeight: 1.7, marginTop: '1rem' }}>
              The low result is evidence of substantial domain shift. It does not establish universal
              cross-domain performance, and it is not comparable to the local 13-class mAP.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Methodology */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container" style={{ maxWidth: '880px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div className="badge">Methodology</div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '2.5rem' }}>Training Pipeline</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem' }}>
              {METHODOLOGY.map((m, i) => (
                <motion.div key={i} initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} className="card">
                  <div style={{ fontSize: '2rem', fontWeight: 900, color: '#DBE2EF', marginBottom: '0.5rem' }}>{m.step}</div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '0.5rem', color: '#112D4E' }}>{m.title}</h4>
                  <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7 }}>{m.body}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
