// ErrorBoundary.jsx
import React from 'react';

export default class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error('Board crashed', error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="error-card">
          <h3>Something went wrong with the board.</h3>
          <p>Try reloading the page or starting a new demo.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
