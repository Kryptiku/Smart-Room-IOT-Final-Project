import './globals.css';

export const metadata = {
  title: 'Smart Room AC Dashboard',
  description: 'Live Firebase dashboard for the smart room aircon simulation.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}