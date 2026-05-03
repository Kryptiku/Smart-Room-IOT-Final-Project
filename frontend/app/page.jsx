'use client';

import { useEffect, useState } from 'react';
import { onValue } from 'firebase/database';
import { roomStateRef } from '../lib/firebase';

const DEFAULT_STATE = {
  occupancy: 0,
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

  const airconOn = Boolean(roomState.airconOn);
  const remainingUntilOn = Math.max(roomState.threshold - roomState.occupancy, 0);

  return (
    <main className="dashboard-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Firebase realtime room monitor</p>
          <h1>Aircon simulation dashboard</h1>
          <p className="lead">
            The live database state turns the virtual AC on once the room reaches the
            configured occupancy threshold.
          </p>
        </div>

        <div className={`status-panel ${airconOn ? 'on' : 'off'}`}>
          <div className="status-label">Current AC state</div>
          <div className="status-value">{roomState.airconStatus}</div>
          <div className="status-subcopy">Threshold: {roomState.threshold} people</div>
        </div>
      </section>

      <section className="grid">
        <article className="metric-card">
          <span className="metric-label">Confirmed occupancy</span>
          <strong className="metric-value">{roomState.occupancy}</strong>
          <span className="metric-footnote">
            {airconOn ? 'Room is over the trigger point.' : `${remainingUntilOn} more to turn on the AC.`}
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