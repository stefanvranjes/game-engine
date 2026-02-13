#include "FuzzyMatcher.h"
#include <numeric>

FuzzyMatcher::MatchResult FuzzyMatcher::Match(const std::string& text, const std::string& pattern, const Config& config) {
    MatchResult result;
    result.matched = false;
    result.score = 0.0f;

    if (pattern.empty()) {
        result.matched = true;
        result.score = 1.0f;
        return result;
    }

    if (text.empty()) {
        return result;
    }

    result.score = CalculateScore(text, pattern, result.positions, config);
    result.matched = result.score >= config.minScore;

    return result;
}

bool FuzzyMatcher::Matches(const std::string& text, const std::string& pattern, bool caseSensitive) {
    Config config;
    config.caseSensitive = caseSensitive;
    return Match(text, pattern, config).matched;
}

float FuzzyMatcher::GetScore(const std::string& text, const std::string& pattern, bool caseSensitive) {
    Config config;
    config.caseSensitive = caseSensitive;
    return Match(text, pattern, config).score;
}

std::string FuzzyMatcher::Transform(const std::string& str, bool toLower) {
    std::string result = str;
    if (toLower) {
        std::transform(result.begin(), result.end(), result.begin(), 
                      [](unsigned char c) { return std::tolower(c); });
    } else {
        std::transform(result.begin(), result.end(), result.begin(), 
                      [](unsigned char c) { return std::toupper(c); });
    }
    return result;
}

std::string FuzzyMatcher::Highlight(const std::string& text, const std::string& pattern, bool caseSensitive) {
    std::vector<int> positions;
    Config config;
    config.caseSensitive = caseSensitive;
    CalculateScore(text, pattern, positions, config);

    if (positions.empty()) {
        return text;
    }

    std::sort(positions.begin(), positions.end());
    std::string result;
    int lastPos = 0;

    for (int pos : positions) {
        if (pos > lastPos) {
            result += text.substr(lastPos, pos - lastPos);
        }
        result += "*" + std::string(1, text[pos]) + "*";
        lastPos = pos + 1;
    }

    if (lastPos < (int)text.length()) {
        result += text.substr(lastPos);
    }

    return result;
}

float FuzzyMatcher::CalculateScore(const std::string& text, const std::string& pattern, 
                                   std::vector<int>& positions, const Config& config) {
    const size_t textLen = text.length();
    const size_t patternLen = pattern.length();

    if (patternLen > textLen) {
        return 0.0f;
    }

    // Convert to lowercase if needed for comparison
    std::string lowerText = config.caseSensitive ? text : Transform(text, true);
    std::string lowerPattern = config.caseSensitive ? pattern : Transform(pattern, true);

    // DP table: dp[i][j] = best score for pattern[0..i] using text[0..j]
    std::vector<std::vector<float>> dp(patternLen + 1, std::vector<float>(textLen + 1, 0.0f));
    std::vector<std::vector<int>> lastMatch(patternLen + 1, std::vector<int>(textLen + 1, -1));

    // Base case: empty pattern matches with perfect score
    for (size_t j = 0; j <= textLen; ++j) {
        dp[0][j] = 1.0f;
    }

    // Fill DP table
    for (size_t i = 1; i <= patternLen; ++i) {
        for (size_t j = i; j <= textLen; ++j) {
            // Option 1: Don't use text[j-1]
            float skipScore = dp[i][j - 1];

            // Option 2: Use text[j-1] to match pattern[i-1]
            float useScore = 0.0f;
            if (CharMatches(lowerText[j - 1], lowerPattern[i - 1], config.caseSensitive)) {
                // Calculate the match score
                float baseScore = dp[i - 1][j - 1];

                // Boost score for consecutive matches
                if (config.highlightConsecutive && j > 1 && lastMatch[i - 1][j - 2] == (int)(j - 2)) {
                    useScore = baseScore + 0.15f;
                }
                // Boost score for word boundary matches
                else if (IsWordBoundary(lowerText, j - 1)) {
                    useScore = baseScore + 0.1f;
                }
                // Standard match
                else {
                    useScore = baseScore + 0.05f;
                }

                if (i == 1 && j == 1) {
                    useScore = std::min(1.0f, baseScore + 0.1f);
                }

                if (i == patternLen && j == textLen) {
                    // Penalize extra characters at the end
                    useScore *= (float)patternLen / textLen;
                }

                if (useScore > 0.0f && i > 0 && j > 0) {
                    lastMatch[i][j] = j - 1;
                }
            }

            // Choose the better option
            if (useScore > skipScore) {
                dp[i][j] = useScore;
                if (i > 0 && j > 0 && CharMatches(lowerText[j - 1], lowerPattern[i - 1], config.caseSensitive)) {
                    lastMatch[i][j] = j - 1;
                }
            } else {
                dp[i][j] = skipScore;
                if (i > 0) {
                    lastMatch[i][j] = lastMatch[i][j - 1];
                }
            }
        }
    }

    // Backtrack to find matched positions
    positions.clear();
    if (dp[patternLen][textLen] > 0.0f) {
        int j = textLen - 1;
        for (int i = patternLen - 1; i >= 0 && j >= 0; --i) {
            // Find the position of pattern[i] in text
            while (j >= 0 && !CharMatches(lowerText[j], lowerPattern[i], config.caseSensitive)) {
                --j;
            }
            if (j >= 0) {
                positions.push_back(j);
                --j;
            }
        }
        std::reverse(positions.begin(), positions.end());
    }

    return std::min(1.0f, dp[patternLen][textLen]);
}

bool FuzzyMatcher::IsWordBoundary(const std::string& text, size_t index) {
    if (index == 0) {
        return true;
    }

    char current = text[index];
    char previous = text[index - 1];

    // Boundary if current is uppercase and previous is lowercase
    if (std::isupper(current) && std::islower(previous)) {
        return true;
    }

    // Boundary if previous is not alphanumeric and current is
    if (!std::isalnum(previous) && std::isalnum(current)) {
        return true;
    }

    // Boundary if it's a digit boundary
    if (std::isdigit(current) && !std::isdigit(previous)) {
        return true;
    }

    return false;
}

bool FuzzyMatcher::CharMatches(char textChar, char patternChar, bool caseSensitive) {
    if (caseSensitive) {
        return textChar == patternChar;
    }
    return std::tolower(textChar) == std::tolower(patternChar);
}
