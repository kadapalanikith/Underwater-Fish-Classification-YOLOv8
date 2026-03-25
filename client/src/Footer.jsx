import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer style={{
      background: '#112D4E',
      color: '#DBE2EF',
      padding: '3rem 0',
      marginTop: '4rem'
    }}>
      <div className="container" style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
          <div>
            <div style={{ fontSize: '1.35rem', fontWeight: 900, letterSpacing: '-0.05em', color: '#fff', marginBottom: '0.75rem' }}>
              DeepSea <span style={{ color: '#3F72AF' }}>AI</span>
            </div>
            <p style={{ fontSize: '0.875rem', color: '#9aafcc', lineHeight: 1.7 }}>
              Real-time underwater fish species classification using YOLOv8 deep learning.
            </p>
          </div>
          <div>
            <div style={{ fontWeight: 700, marginBottom: '1rem', color: '#fff', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Navigate</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {[['/', 'Home'], ['/demo', 'Demo'], ['/research', 'Research'], ['/contact', 'Contact']].map(([to, label]) => (
                <Link key={to} to={to} style={{ color: '#9aafcc', textDecoration: 'none', fontSize: '0.875rem', transition: 'color 0.2s' }}
                  onMouseEnter={e => e.target.style.color = '#DBE2EF'}
                  onMouseLeave={e => e.target.style.color = '#9aafcc'}
                >
                  {label}
                </Link>
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 700, marginBottom: '1rem', color: '#fff', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Technologies</div>
            {['YOLOv8 (Ultralytics)', 'FastAPI', 'React + Vite', 'PyTorch'].map(t => (
              <div key={t} style={{ color: '#9aafcc', fontSize: '0.875rem', marginBottom: '0.4rem' }}>{t}</div>
            ))}
          </div>
        </div>
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1.5rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ color: '#5a7499', fontSize: '0.8rem' }}>© 2026 DeepSea AI. Academic Research Project.</div>
          <div style={{ color: '#5a7499', fontSize: '0.8rem' }}>Built with YOLOv8 • FastAPI • React</div>
        </div>
      </div>
    </footer>
  );
}
