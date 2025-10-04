import { useNavigate } from 'react-router-dom';
import { Scale, Gavel, Shield, ChevronRight } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white overflow-hidden">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMzLjMxNCAwIDYgMi42ODYgNiA2cy0yLjY4NiA2LTYgNi02LTIuNjg2LTYtNiAyLjY4Ni02IDYtNiIgc3Ryb2tlPSJyZ2JhKDI1NSwgMjU1LCAyNTUsIDAuMDMpIi8+PC9nPjwvc3ZnPg==')] opacity-20"></div>

      <nav className="relative z-10 px-6 py-6 flex justify-between items-center max-w-7xl mx-auto">
        <div className="flex items-center gap-3 animate-fade-in">
          <Scale className="w-8 h-8 text-amber-400" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-amber-200 bg-clip-text text-transparent">
            Apna Waqeel
          </h1>
        </div>
        <button
          onClick={() => navigate('/chat')}
          className="px-6 py-2 bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-amber-500/50"
        >
          Get Started
        </button>
      </nav>

      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32">
        <div className="flex flex-col lg:flex-row items-center gap-16">
          <div className="flex-1 space-y-8 animate-slide-up">
            <div className="inline-block px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-full text-amber-400 text-sm font-medium animate-fade-in">
              AI-Powered Legal Assistant
            </div>

            <h2 className="text-5xl lg:text-7xl font-bold leading-tight animate-fade-in-delay-1">
              Your Personal
              <span className="block bg-gradient-to-r from-amber-400 via-amber-300 to-amber-500 bg-clip-text text-transparent">
                Legal Companion
              </span>
            </h2>

            <p className="text-xl text-slate-300 leading-relaxed max-w-2xl animate-fade-in-delay-2">
              Get instant answers to your legal questions with AI-powered precision. Navigate complex legal matters with confidence and clarity.
            </p>

            <button
              onClick={() => navigate('/chat')}
              className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 text-lg font-bold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-amber-500/50 animate-fade-in-delay-3"
            >
              Start Legal Chat
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          <div className="flex-1 relative animate-float">
            <div className="relative w-full max-w-lg mx-auto">
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500 to-blue-500 rounded-full blur-3xl opacity-20 animate-pulse-slow"></div>

              <div className="relative bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-12 shadow-2xl">
                <div className="flex justify-center mb-8">
                  <div className="relative">
                    <div className="absolute inset-0 bg-amber-500 rounded-full blur-xl opacity-50 animate-pulse"></div>
                    <Scale className="relative w-24 h-24 text-amber-400" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-700/50 hover:border-amber-500/50 transition-all duration-300 hover:scale-105 animate-fade-in-delay-1">
                    <Gavel className="w-8 h-8 text-amber-400 mb-3" />
                    <h3 className="font-semibold mb-2">Expert Guidance</h3>
                    <p className="text-sm text-slate-400">Professional legal insights</p>
                  </div>

                  <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-700/50 hover:border-amber-500/50 transition-all duration-300 hover:scale-105 animate-fade-in-delay-2">
                    <Shield className="w-8 h-8 text-amber-400 mb-3" />
                    <h3 className="font-semibold mb-2">Secure & Private</h3>
                    <p className="text-sm text-slate-400">Your data is protected</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8 animate-fade-in-delay-4">
          <div className="bg-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 hover:border-amber-500/50 transition-all duration-300 hover:scale-105">
            <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
              <Scale className="w-6 h-6 text-amber-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">Instant Answers</h3>
            <p className="text-slate-400">Get immediate responses to your legal queries, anytime, anywhere.</p>
          </div>

          <div className="bg-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 hover:border-amber-500/50 transition-all duration-300 hover:scale-105">
            <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
              <Gavel className="w-6 h-6 text-amber-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">Verified Sources</h3>
            <p className="text-slate-400">All responses backed by legal references and citations.</p>
          </div>

          <div className="bg-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 hover:border-amber-500/50 transition-all duration-300 hover:scale-105">
            <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-amber-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">24/7 Available</h3>
            <p className="text-slate-400">Access legal assistance whenever you need it, day or night.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
