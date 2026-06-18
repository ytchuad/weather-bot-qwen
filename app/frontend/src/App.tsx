import { Routes, Route, Navigate } from "react-router-dom"
import Layout from "./components/Layout"
import Hub from "./pages/Hub"
import Strategies from "./pages/Strategies"

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Hub />} />
        <Route path="/strategies" element={<Strategies />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
