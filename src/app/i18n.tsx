// 画面の表示言語(日本語/英語)。2026-10-13追加(英語UI)。
// 漣さんとの確認事項: ①OSの言語で自動判定し、画面右上のボタンで手動切替(選択は次回も保持)
// ②Python側のエラー文も英語化(言語は環境変数 AIDE_LANG で子プロセスに渡す。electron/main.ts参照)
// ③Steeringのプリセット文例も英語版を用意する(python/presets/steering_presets.json の "en")
//
// 文言は辞書のキーではなく、使う場所に t("日本語", "English") の形で並べて書く
// (どの画面の文言かがコードを読めばすぐ分かり、片方だけ書き忘れることも型で防げる)。
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Lang = "ja" | "en";
export type T = (ja: string, en: string) => string;

const STORAGE_KEY = "aide.lang";

function detectLang(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "ja" || saved === "en") return saved;
  } catch {
    // localStorageが使えない環境ではOSの言語で決める
  }
  return navigator.language.toLowerCase().startsWith("ja") ? "ja" : "en";
}

type LangContextValue = { lang: Lang; setLang: (lang: Lang) => void; t: T };

const LangContext = createContext<LangContextValue>({
  lang: "ja",
  setLang: () => {},
  t: (ja) => ja,
});

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectLang);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // 保存できなくても今回の表示は切り替わる
    }
  }, []);

  // Python子プロセスのエラー文の言語も合わせる(起動するたびに環境変数で渡される)
  useEffect(() => {
    document.documentElement.lang = lang;
    window.electronAPI?.setLang?.(lang);
  }, [lang]);

  const value = useMemo<LangContextValue>(
    () => ({ lang, setLang, t: (ja, en) => (lang === "en" ? en : ja) }),
    [lang, setLang]
  );
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang(): LangContextValue {
  return useContext(LangContext);
}
