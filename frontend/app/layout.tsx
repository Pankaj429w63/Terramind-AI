import "./globals.css";

export const metadata = {
  title: "TerraMind AI",
  description: "AI for smarter agriculture",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
