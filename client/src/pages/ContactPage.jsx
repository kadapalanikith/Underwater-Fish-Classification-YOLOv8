import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Mail, Phone, MapPin, GitFork, GraduationCap, Users, BookOpen, Send } from 'lucide-react';

const TEAM = [
  {
    initials: 'NK',
    name: 'Nikith Kadapalaneni',
    role: 'Team Member',
    description: 'Computer Vision & Full-Stack Developer. Responsible for model training, FastAPI backend, and React frontend.',
    contact: { email: 'nikith@example.com', linkedin: 'linkedin.com/in/nikith', github: 'github.com/kadapalanikith' },
    isGuide: false,
  },
  {
    initials: 'TS',
    name: 'Team Member 2',
    role: 'Team Member',
    description: 'AI Researcher specializing in dataset curation, annotation, and benchmarking of deep learning models.',
    contact: { email: 'member2@example.com', linkedin: 'linkedin.com/in/member2', github: 'github.com/member2' },
    isGuide: false,
  },
  {
    initials: 'DR',
    name: 'Dr. Research Guide',
    role: 'Project Guide',
    description: 'Associate Professor, Department of Computer Science. Mentor for AI research, faculty advisor for IEEE paper submission.',
    contact: { email: 'guide@university.edu', linkedin: 'linkedin.com/in/drguide' },
    isGuide: true,
  },
];

const fadeIn = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.55 } } };

export default function ContactPage() {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [sent, setSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // In production this would POST to a contact endpoint
    setSent(true);
    setTimeout(() => setSent(false), 4000);
    setFormData({ name: '', email: '', message: '' });
  };

  return (
    <div style={{ paddingTop: '70px' }}>

      {/* Header */}
      <section style={{ padding: '4rem 0 3rem', background: 'linear-gradient(135deg, #F9F7F7 0%, #DBE2EF 100%)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="badge">Contact Us</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3rem)', marginBottom: '1rem' }}>Meet the Team</h1>
          <p style={{ color: '#6b7f9e', maxWidth: '540px', margin: '0 auto', fontSize: '1.05rem' }}>
            Get in touch with us for research collaboration, model inquiries, or project feedback.
          </p>
        </div>
      </section>

      {/* Team Cards */}
      <section className="section" style={{ background: '#fff' }}>
        <div className="container">
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn} style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
            <div className="badge"><Users size={14} style={{ display: 'inline', marginRight: 4 }} />Project Team</div>
            <h2 style={{ fontSize: '1.8rem' }}>The People Behind DeepSea AI</h2>
          </motion.div>
          <div className="contact-cards">
            {TEAM.map((member, i) => (
              <motion.div key={i} initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}
                className="contact-card" style={{ position: 'relative' }}>
                {member.isGuide && (
                  <div style={{
                    position: 'absolute', top: 16, right: 16,
                    background: 'rgba(17,45,78,0.08)',
                    color: '#112D4E', padding: '0.2rem 0.6rem',
                    borderRadius: 9999, fontSize: '0.7rem', fontWeight: 700,
                    textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 4
                  }}>
                    <GraduationCap size={12} /> Guide
                  </div>
                )}
                <div className="contact-avatar" style={member.isGuide ? { background: '#112D4E', color: '#DBE2EF', borderColor: '#112D4E' } : {}}>
                  {member.initials}
                </div>
                <h3 style={{ fontWeight: 800, fontSize: '1.1rem', marginBottom: '0.25rem', color: '#112D4E' }}>{member.name}</h3>
                <span className={`contact-role ${member.isGuide ? 'guide-role' : ''}`}>{member.role}</span>
                <p style={{ fontSize: '0.875rem', color: '#6b7f9e', lineHeight: 1.7, marginBottom: '1.25rem' }}>{member.description}</p>
                <div className="contact-links">
                  {member.contact.email && (
                    <a href={`mailto:${member.contact.email}`} className="contact-link">
                      <Mail size={13} /> Email
                    </a>
                  )}
                  {member.contact.linkedin && (
                    <a href={`https://${member.contact.linkedin}`} target="_blank" rel="noreferrer" className="contact-link">
                      <Phone size={13} /> LinkedIn
                    </a>
                  )}
                  {member.contact.github && (
                    <a href={`https://${member.contact.github}`} target="_blank" rel="noreferrer" className="contact-link">
                      <GitFork size={13} /> GitHub
                    </a>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Form */}
      <section className="section" style={{ background: '#F9F7F7' }}>
        <div className="container" style={{ maxWidth: '680px' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={fadeIn}>
            <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
              <div className="badge">Send a Message</div>
              <h2 style={{ fontSize: '1.8rem' }}>Get In Touch</h2>
              <p style={{ color: '#6b7f9e', marginTop: '0.5rem' }}>Have a question or collaboration idea? We'd love to hear from you.</p>
            </div>
            <div className="card" style={{ padding: '2.5rem' }}>
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {[
                  { name: 'name',    label: 'Your Name',    type: 'text',  placeholder: 'John Doe' },
                  { name: 'email',   label: 'Email Address', type: 'email', placeholder: 'john@email.com' },
                ].map(field => (
                  <div key={field.name}>
                    <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem', color: '#112D4E' }}>
                      {field.label}
                    </label>
                    <input
                      type={field.type}
                      required
                      placeholder={field.placeholder}
                      value={formData[field.name]}
                      onChange={e => setFormData({ ...formData, [field.name]: e.target.value })}
                      style={{
                        width: '100%', padding: '0.8rem 1rem',
                        border: '1.5px solid #DBE2EF', borderRadius: 12,
                        fontSize: '0.9rem', outline: 'none', fontFamily: 'inherit',
                        background: '#F9F7F7', color: '#112D4E',
                        transition: 'border-color 0.2s',
                      }}
                      onFocus={e => e.target.style.borderColor = '#3F72AF'}
                      onBlur={e => e.target.style.borderColor = '#DBE2EF'}
                    />
                  </div>
                ))}
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.5rem', color: '#112D4E' }}>Message</label>
                  <textarea
                    required
                    placeholder="Your message..."
                    rows={5}
                    value={formData.message}
                    onChange={e => setFormData({ ...formData, message: e.target.value })}
                    style={{
                      width: '100%', padding: '0.8rem 1rem',
                      border: '1.5px solid #DBE2EF', borderRadius: 12,
                      fontSize: '0.9rem', outline: 'none', fontFamily: 'inherit',
                      background: '#F9F7F7', color: '#112D4E', resize: 'vertical',
                      transition: 'border-color 0.2s',
                    }}
                    onFocus={e => e.target.style.borderColor = '#3F72AF'}
                    onBlur={e => e.target.style.borderColor = '#DBE2EF'}
                  />
                </div>
                <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '1rem', fontSize: '1rem' }}>
                  {sent ? '✓ Message Sent!' : <><Send size={18} /> Send Message</>}
                </button>
              </form>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Academic Info */}
      <section className="section" style={{ background: '#112D4E', color: '#DBE2EF', borderBottom: 'none' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
            <BookOpen size={28} color="#3F72AF" />
            <h2 style={{ color: '#fff', fontSize: '1.6rem' }}>Academic Affiliation</h2>
          </div>
          <p style={{ color: '#9aafcc', maxWidth: '500px', margin: '0 auto', fontSize: '0.95rem', lineHeight: 1.7 }}>
            This project is affiliated with the Department of Computer Science &amp; Engineering and is targeting IEEE publication in the domain of marine AI and computer vision.
          </p>
        </div>
      </section>
    </div>
  );
}
