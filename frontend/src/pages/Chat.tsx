import { useState, useRef, useEffect } from 'react';
import { Scale, Send, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, UserButton } from '@clerk/clerk-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  html?: string;
  references?: Array<{ title: string; url: string }>;
}

export default function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // --- helper: strip header token if present ---
  const HEADER_TOKEN = '<|start_header_id|>assistant<|end_header_id|>';
  const stripHeaderToken = (text: string) =>
    text.startsWith(HEADER_TOKEN) ? text.slice(HEADER_TOKEN.length).trim() : text;

  // --- helper: escape HTML to avoid XSS ---
  const escapeHtml = (str: string) =>
    str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');

  // --- helper: inline formatting (code, bold, italic) ---
  const processInline = (s: string) =>
    s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>');

  // --- lightweight markdown -> HTML converter (handles headings, lists, paragraphs, bold, italic, links, inline code) ---
  const mdToHtml = (md: string) => {
    // escape first to avoid raw HTML injection
    const escaped = escapeHtml(md);

    // convert Markdown links [text](url) -> safe anchor tags
    const withLinks = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, text, url) => {
      // escape text and url separately; url kept safe for href attribute
      const safeText = processInline(text);
      const safeUrl = escapeHtml(url);
      return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeText}</a>`;
    });

    const lines = withLinks.split(/\r?\n/);
    let html = '';
    let inList = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // headings
      const h1 = line.match(/^#\s+(.*)/);
      const h2 = line.match(/^##\s+(.*)/);
      const h3 = line.match(/^###\s+(.*)/);

      if (h1) {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        html += `<h1>${processInline(h1[1])}</h1>`;
        continue;
      }
      if (h2) {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        html += `<h2>${processInline(h2[1])}</h2>`;
        continue;
      }
      if (h3) {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        html += `<h3>${processInline(h3[1])}</h3>`;
        continue;
      }

      // unordered list item (supports *, -, +)
      const ulMatch = line.match(/^(\*|-|\+)\s+(.*)/);
      if (ulMatch) {
        if (!inList) {
          html += '<ul>';
          inList = true;
        }
        html += `<li>${processInline(ulMatch[2])}</li>`;
        continue;
      }

      // blank line -> close list / paragraph separation
      if (line === '') {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        continue;
      }

      // normal paragraph line
      if (inList) {
        html += '</ul>';
        inList = false;
      }
      html += `<p>${processInline(line)}</p>`;
    }

    if (inList) html += '</ul>';
    return html;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
    };

    // Add the user's message to UI immediately
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Format chat history - make sure it's the right format for Flask
      const previousMessages = [...messages]; // Don't include current message
      
      // Take only last 5 exchanges (10 messages) in pairs
      // For a history like: [msg1, response1, msg2, response2, msg3, response3]
      // We want to keep the chronological order intact
      const maxHistoryLength = 10; // 5 exchanges
      const historyToSend = previousMessages.slice(-maxHistoryLength);
      
      // Create a clean history array with just role and content
      const chatHistoryForBackend = historyToSend.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Create payload with this history
      const payload = {
        question: input,
        chat_history: chatHistoryForBackend
      };

      // Enhanced debug logging
      console.log('[CHAT] Sending question:', input);
      console.log('[CHAT] Chat history length:', chatHistoryForBackend.length);
      console.log('[CHAT] Full payload:', JSON.stringify(payload, null, 2));
      
      if (chatHistoryForBackend.length > 0) {
        console.log('[CHAT] First history item:', JSON.stringify(chatHistoryForBackend[0]));
        console.log('[CHAT] Last history item:', JSON.stringify(chatHistoryForBackend[chatHistoryForBackend.length - 1]));
      }

      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      console.log('[CHAT] Response status:', response.status, response.statusText);

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        console.error('[CHAT] Error response body:', text);
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      console.log('[CHAT] Backend response type:', typeof data);
      console.log('[CHAT] Backend returned keys:', Object.keys(data));
      console.log('[CHAT] Chat response preview:', 
        typeof data.chat_response === 'string' 
          ? data.chat_response.substring(0, 100) + '...' 
          : 'Not a string');

      // ensure chat_response is string
      const rawResponse =
        typeof data.chat_response === 'string' ? data.chat_response : String(data.chat_response);

      // strip header token if present and convert markdown -> safe HTML
      const cleaned = stripHeaderToken(rawResponse);
      const html = mdToHtml(cleaned);

      const assistantMessage: Message = {
        role: 'assistant',
        content: cleaned,
        html,
        references: data.references || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('[CHAT] Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex flex-col">
      <nav className="bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 px-6 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <Scale className="w-7 h-7 text-amber-400" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-amber-400 to-amber-200 bg-clip-text text-transparent">
              Apna Waqeel – Legal Assistant
            </h1>
          </button>
          
          {/* Add authentication UI components */}
          <div className="flex items-center gap-4">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-900 rounded-lg font-medium transition-all">
                  Sign In
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </div>
      </nav>

      <SignedIn>
        {/* Existing chat UI remains the same */}
        <div className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 overflow-hidden flex gap-6">
          <div className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-6 mb-6 pr-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center space-y-4 animate-fade-in">
                    <div className="inline-block p-6 bg-amber-500/10 rounded-full">
                      <Scale className="w-16 h-16 text-amber-400" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">Ask Your Legal Question</h2>
                    <p className="text-slate-400 max-w-md">
                      Type your question below and get instant legal guidance with verified references.
                    </p>
                  </div>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    } animate-slide-up`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-6 py-4 ${
                        message.role === 'user'
                          ? 'bg-gradient-to-r from-amber-500 to-amber-400 text-slate-900'
                          : 'bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 text-white'
                      }`}
                    >
                      {message.html ? (
                        <div
                          className="whitespace-pre-wrap leading-relaxed"
                          // html produced by mdToHtml is escaped then converted -> safe for rendering
                          dangerouslySetInnerHTML={{ __html: message.html }}
                        />
                      ) : (
                        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                      )}

                      {message.references && message.references.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-2">
                          <p className="text-sm font-semibold text-amber-400">References:</p>
                          {message.references.map((ref, refIndex) => (
                            <a
                              key={refIndex}
                              href={ref.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-2 text-sm text-slate-300 hover:text-amber-400 transition-colors group"
                            >
                              <ExternalLink className="w-4 h-4 flex-shrink-0 group-hover:scale-110 transition-transform" />
                              <span className="underline">{ref.title}</span>
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}

              {isLoading && (
                <div className="flex justify-start animate-slide-up">
                  <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl px-6 py-4">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="relative">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask your legal question..."
                  disabled={isLoading}
                  className="flex-1 bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-xl px-6 py-4 text-white placeholder-slate-500 focus:outline-none focus:border-amber-500/50 focus:ring-2 focus:ring-amber-500/20 transition-all disabled:opacity-50"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="px-6 py-4 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 hover:shadow-lg hover:shadow-amber-500/50 group"
                >
                  <Send className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
                </button>
              </div>
            </div>
          </div>

          {messages.some((msg) => msg.references && msg.references.length > 0) && (
            <div className="hidden lg:block w-80 animate-slide-in-right">
              <div className="sticky top-24 bg-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                  <ExternalLink className="w-5 h-5 text-amber-400" />
                  Legal References
                </h3>
                <div className="space-y-3">
                  {messages
                    .filter((msg) => msg.references && msg.references.length > 0)
                    .flatMap((msg) => msg.references!)
                    .map((ref, index) => (
                      <a
                        key={index}
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block p-4 bg-slate-900/50 rounded-lg border border-slate-700/50 hover:border-amber-500/50 transition-all group hover:scale-105"
                      >
                        <p className="text-sm font-medium text-white group-hover:text-amber-400 transition-colors">
                          {ref.title}
                        </p>
                        <p className="text-xs text-slate-500 mt-1 truncate">{ref.url}</p>
                      </a>
                    ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </SignedIn>
      
      <SignedOut>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center p-8 max-w-md">
            <Scale className="w-16 h-16 text-amber-400 mx-auto mb-6" />
            <h2 className="text-2xl font-bold text-white mb-4">Sign in to Access Legal Assistance</h2>
            <p className="text-slate-400 mb-6">
              Please sign in to get answers to your legal questions and access professional legal guidance.
            </p>
            <SignInButton mode="modal">
              <button className="px-6 py-3 bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-slate-900 rounded-xl font-semibold transition-all">
                Sign In to Continue
              </button>
            </SignInButton>
          </div>
        </div>
      </SignedOut>
    </div>
  );
}
