import { useEffect } from "react";
import { AlertTriangle, Bell, ShieldCheck, Siren } from "lucide-react";
import { COLORS } from "../format.js";

const ICON = { Alert: Siren, Warning: AlertTriangle, Safe: ShieldCheck };

function Toast({ toast, dismiss }) {
  useEffect(() => {
    const t = setTimeout(() => dismiss(toast.key), 6000);
    return () => clearTimeout(t);
  }, [toast.key, dismiss]);
  const Icon = ICON[toast.level] ?? Bell;
  const color = COLORS[toast.level] ?? "var(--info)";
  return (
    <div className="toast" style={{ borderLeftColor: color }} onClick={() => dismiss(toast.key)}>
      <Icon size={18} color={color} />
      {toast.title}
    </div>
  );
}

export default function Toasts({ toasts, dismiss }) {
  return (
    <div className="toasts" role="status">
      {toasts.map((t) => (
        <Toast key={t.key} toast={t} dismiss={dismiss} />
      ))}
    </div>
  );
}
