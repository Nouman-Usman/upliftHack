import { useState, useRef, useEffect } from 'react';
import { Scale, Send, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';

interface Message {
  role: 'user' | 'assistant';
  content: string;
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

  // using react-markdown + remark-gfm + rehype-sanitize for safe markdown rendering

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
      const chatHistory = [...messages, userMessage].map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch('https://nouman-usman-flask--5000.prod1a.defang.dev/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          chat_history: chatHistory,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // ensure chat_response is string
      const rawResponse =
        typeof data.chat_response === 'string' ? data.chat_response : String(data.chat_response);

      // strip header token if present and keep markdown string
      const cleaned = stripHeaderToken(rawResponse);

      const assistantMessage: Message = {
        role: 'assistant',
        content: cleaned,
        references: data.references || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
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
        </div>
      </nav>

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
                    {message.role === 'assistant' ? (
                      <div className="whitespace-pre-wrap leading-relaxed prose prose-invert">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeSanitize]}
                          components={{
                            a: ({ node, ...props }) => (
                              <a {...props} target="_blank" rel="noopener noreferrer" />
                            ),
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
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
    </div>
  );
}
