-- | A minimal JSON parser over lazy ByteStrings. Zero dependencies beyond
-- GHC boot libraries (bytestring, base). Sufficient for safetensors headers
-- and model config files; not a general-purpose JSON library.
module HQ.JSON
  ( JValue(..)
  , parseJSON
  , lookupKey
  , jToFloat
  , jToInt
  , jToString
  , jToArray
  ) where

import qualified Data.ByteString.Lazy.Char8 as LC
import Data.Char (chr, digitToInt, isDigit, isHexDigit)
import Data.List (foldl')

data JValue
  = JNull
  | JBool !Bool
  | JNum !Double
  | JStr !String
  | JArr [JValue]
  | JObj [(String, JValue)]
  deriving (Eq, Show)

-- | Parse a complete JSON document; fails on trailing garbage.
parseJSON :: LC.ByteString -> Either String JValue
parseJSON bs = do
  (v, rest) <- jvalue (skipWs bs)
  if LC.all isSpaceChar rest
    then Right v
    else Left "trailing garbage after JSON value"

isSpaceChar :: Char -> Bool
isSpaceChar c = c == ' ' || c == '\t' || c == '\n' || c == '\r'

skipWs :: LC.ByteString -> LC.ByteString
skipWs = LC.dropWhile isSpaceChar

jvalue :: LC.ByteString -> Either String (JValue, LC.ByteString)
jvalue bs = case LC.uncons bs of
  Nothing -> Left "unexpected end of input"
  Just (c, rest) -> case c of
    '{' -> jobject rest
    '[' -> jarray rest
    '"' -> do (s, r) <- jstring rest
              pure (JStr s, r)
    't' -> lit bs "true" (JBool True)
    'f' -> lit bs "false" (JBool False)
    'n' -> lit bs "null" JNull
    _ | c == '-' || isDigit c -> jnum bs
      | otherwise -> Left ("unexpected character " ++ [c])
  where
    lit b want v =
      let (tok, r) = LC.splitAt (fromIntegral (length want)) b
      in if LC.unpack tok == want
           then Right (v, r)
           else Left ("expected " ++ want)

jobject :: LC.ByteString -> Either String (JValue, LC.ByteString)
jobject rest0 = do
  let rest = skipWs rest0
  case LC.uncons rest of
    Just ('}', r) -> pure (JObj [], r)
    _ -> loop rest [] 
  where
    loop b acc = do
      (k, b1) <- case LC.uncons (skipWs b) of
        Just ('"', r) -> jstring r
        _ -> Left "expected object key"
      b1' <- case LC.uncons (skipWs b1) of
        Just (':', r) -> Right r
        _ -> Left "expected : after object key"
      (v, b2) <- jvalue (skipWs b1')
      b3 <- case LC.uncons (skipWs b2) of
        Just (',', r) -> Right r
        Just ('}', r) -> Right r
        _ -> Left "expected , or } in object"
      let acc' = (k, v) : acc
      case LC.uncons (skipWs b2) of
        Just ('}', r) -> pure (JObj (reverse acc'), r)
        _ -> loop b3 acc'

jarray :: LC.ByteString -> Either String (JValue, LC.ByteString)
jarray rest0 = do
  let rest = skipWs rest0
  case LC.uncons rest of
    Just (']', r) -> pure (JArr [], r)
    _ -> loop rest []
  where
    loop b acc = do
      (v, b1) <- jvalue (skipWs b)
      let acc' = v : acc
      case LC.uncons (skipWs b1) of
        Just (',', r) -> loop r acc'
        Just (']', r) -> pure (JArr (reverse acc'), r)
        _ -> Left "expected , or ] in array"

-- | Parse a string literal; input starts AFTER the opening quote.
jstring :: LC.ByteString -> Either String (String, LC.ByteString)
jstring = go id
  where
    go acc b = case LC.uncons b of
      Nothing -> Left "unterminated string"
      Just ('"', r) -> Right (acc [], r)
      Just ('\\', r) -> escape r (\s r' -> go (acc . (s ++)) r')
      Just (c, r)
        | c <= '\x1f' -> Left "control char in string"
        | otherwise -> go (acc . (c:)) r
    escape b k = case LC.uncons b of
      Nothing -> Left "unterminated escape"
      Just (c, r) -> case c of
        '"'  -> k "\"" r
        '\\' -> k "\\" r
        '/'  -> k "/" r
        'b'  -> k "\b" r
        'f'  -> k "\f" r
        'n'  -> k "\n" r
        'r'  -> k "\r" r
        't'  -> k "\t" r
        'u'  ->
          let hex4 = LC.take 4 r
              rest = LC.drop 4 r
          in if LC.length hex4 == 4 && LC.all isHexDigit hex4
               then
                 let cp = foldl' (\a w -> a * 16 + digitToInt w) 0 (LC.unpack hex4)
                 in k [chr cp] rest
               else Left "bad \\u escape"
        _ -> Left ("bad escape \\" ++ [c])

-- | Parse a number (int or double).
jnum :: LC.ByteString -> Either String (JValue, LC.ByteString)
jnum b =
  let (tok, r) = LC.span (\c -> isDigit c || c `elem` ("-+.eE" :: String)) b
      s = LC.unpack tok
  in case reads s :: [(Double, String)] of
       [(d, "")] -> Right (JNum d, r)
       _ -> Left ("bad number: " ++ s)

--------------------------------------------------------------------------------
-- Accessors
--------------------------------------------------------------------------------

-- | Look up a key in a JSON object (top level).
lookupKey :: String -> JValue -> Maybe JValue
lookupKey k (JObj kvs) = lookup k kvs
lookupKey _ _ = Nothing

jToFloat :: JValue -> Maybe Double
jToFloat (JNum d) = Just d
jToFloat _ = Nothing

jToInt :: JValue -> Maybe Int
jToInt (JNum d) = Just (round d)
jToInt _ = Nothing

jToString :: JValue -> Maybe String
jToString (JStr s) = Just s
jToString _ = Nothing

jToArray :: JValue -> Maybe [JValue]
jToArray (JArr xs) = Just xs
jToArray _ = Nothing
