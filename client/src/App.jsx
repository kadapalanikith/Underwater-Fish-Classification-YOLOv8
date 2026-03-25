import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './Navbar.jsx';
import Footer from './Footer.jsx';
import HomePage from './pages/HomePage.jsx';
import DemoPage from './pages/DemoPage.jsx';
import ResearchPage from './pages/ResearchPage.jsx';
import ContactPage from './pages/ContactPage.jsx';

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <Routes>
          <Route path="/"         element={<HomePage />}    />
          <Route path="/demo"     element={<DemoPage />}    />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/contact"  element={<ContactPage />}  />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
