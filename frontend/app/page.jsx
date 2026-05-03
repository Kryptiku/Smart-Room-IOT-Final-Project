  'use client';

import { useEffect, useState } from 'react';
import { onValue } from 'firebase/database';
import { roomStateRef } from '../lib/firebase';

const DEFAULT_STATE = {
  occupancy: 0,
  lightsOn: false,
  lightsStatus: 'OFF',
  fansOn: false,
  fansStatus: 'OFF',
  airconOn: false,
  airconStatus: 'OFF',
  threshold: 5,
  updatedAt: null,
};

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return 'Waiting for the first update';
  }

  return new Date(timestamp * 1000).toLocaleString();
}

export default function DashboardPage() {
  const [roomState, setRoomState] = useState(DEFAULT_STATE);
  const [connectionState, setConnectionState] = useState('Connecting to Firebase...');

  useEffect(() => {
    const unsubscribe = onValue(
      roomStateRef,
      (snapshot) => {
        const value = snapshot.val() || {};
        setRoomState({
          ...DEFAULT_STATE,
          ...value,
        });
        setConnectionState('Live');
      },
      (error) => {
        setConnectionState(`Database error: ${error.message}`);
      },
    );

    return () => unsubscribe();
  }, []);

  const lightsOn = Boolean(roomState.lightsOn);
  const fansOn = Boolean(roomState.fansOn);
  const airconOn = Boolean(roomState.airconOn);

  return (
    <main className="dashboard-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Firebase realtime room monitor</p>
          <h1>Smart Room Control Dashboard</h1>
        </div>

        <div className="appliance-status-grid">
          <div className={`status-panel ${lightsOn ? 'on' : 'off'}`}>
            <div className="status-label">Lights</div>
            <div className="status-value">{roomState.lightsStatus}</div>
            <div className="status-subcopy">Occupancy ≥ 1</div>
          </div>

          <div className={`status-panel ${fansOn ? 'on' : 'off'}`}>
            <div className="status-label">Fans</div>
            <div className="status-value">{roomState.fansStatus}</div>
            <div className="status-subcopy">Occupancy ≥ 2</div>
          </div>

          <div className={`status-panel ${airconOn ? 'on' : 'off'}`}>
            <div className="status-label">Air Conditioner</div>
            <div className="status-value">{roomState.airconStatus}</div>
            <div className="status-subcopy">Occupancy ≥ 3</div>
          </div>
        </div>
      </section>

      <section className="grid">
        <article className="metric-card">
          <span className="metric-label">Confirmed occupancy</span>
          <strong className="metric-value">{roomState.occupancy}</strong>
          <span className="metric-footnote">
            {roomState.occupancy >= 5 
              ? 'All appliances are active.' 
              : `${roomState.occupancy >= 3 
                ? 'Lights and fans are on.' 
                : roomState.occupancy >= 1 
                ? 'Lights are on.' 
                : 'No appliances active.'}`}
          </span>
        </article>

        <article className="metric-card">
          <span className="metric-label">Realtime connection</span>
          <strong className="metric-value metric-value--small">{connectionState}</strong>
          <span className="metric-footnote">Updates stream in automatically with Firebase listeners.</span>
        </article>

        <article className="metric-card metric-card--wide">
          <span className="metric-label">Last database update</span>
          <strong className="metric-value metric-value--small">{formatTimestamp(roomState.updatedAt)}</strong>
          <span className="metric-footnote">The dashboard re-renders the moment the database changes.</span>
        </article>
      </section>
    </main>
  );
}