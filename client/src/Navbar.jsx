import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <nav>
      <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '70px', gap: '2rem' }}>
        <Link to="/" className="nav-logo" style={{ minWidth: 'max-content' }}>
          DeepSea <span>AI</span>
        </Link>
        <div className="nav-links" style={{ display: 'flex', gap: '1.5rem', flexWrap: 'nowrap' }}>
          <Link to="/" className={`nav-link ${pathname === '/' ? 'active' : ''}`}>Home</Link>
          <Link to="/demo" className={`nav-link ${pathname === '/demo' ? 'active' : ''}`}>Demo</Link>
          <Link to="/research" className={`nav-link ${pathname === '/research' ? 'active' : ''}`}>Research</Link>
          <Link to="/contact" className={`nav-link ${pathname === '/contact' ? 'active' : ''}`}>Contact</Link>
          <Link to="/demo" className="nav-btn">Try Model →</Link>
        </div>
      </div>
    </nav>
  );
}
