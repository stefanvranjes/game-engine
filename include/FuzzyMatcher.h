#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

/**
 * @brief Fuzzy matching utility for advanced search functionality
 * 
 * Implements fuzzy string matching with scoring to enable
 * flexible searching with typo tolerance and partial matches.
 * 
 * Examples:
 * - "SpriteRenderer" matches "SprRnd" (score: 0.8)
 * - "PlayerController" matches "PlyrCtrl" (score: 0.9)
 * - "CreateGameObject" matches "cgo" (score: 0.7)
 */
class FuzzyMatcher {
public:
    /**
     * @brief Result of a fuzzy match operation
     */
    struct MatchResult {
        bool matched;           // Did the search match?
        float score;            // Match score (0.0 - 1.0)
        std::vector<int> positions;  // Positions of matched characters
    };

    /**
     * @brief Configuration for fuzzy matching
     */
    struct Config {
        bool caseSensitive = false;        // Case-sensitive matching
        bool wholeWordOnly = false;        // Only match whole words
        float minScore = 0.0f;             // Minimum score threshold (0.0-1.0)
        bool highlightConsecutive = true;  // Boost score for consecutive characters
    };

    /**
     * @brief Perform fuzzy matching
     * @param text The text to search in
     * @param pattern The search pattern (fuzzy)
     * @param config Matching configuration
     * @return MatchResult with match status and score
     */
    static MatchResult Match(const std::string& text, const std::string& pattern, const Config& config = Config());

    /**
     * @brief Simple boolean fuzzy match check
     * @param text Text to search in
     * @param pattern Pattern to search for
     * @param caseSensitive Whether to be case-sensitive
     * @return true if pattern fuzzy-matches text
     */
    static bool Matches(const std::string& text, const std::string& pattern, bool caseSensitive = false);

    /**
     * @brief Get match score between 0.0 and 1.0
     * @param text Text to search in
     * @param pattern Pattern to search for
     * @param caseSensitive Whether to be case-sensitive
     * @return Score from 0.0 (no match) to 1.0 (perfect match)
     */
    static float GetScore(const std::string& text, const std::string& pattern, bool caseSensitive = false);

    /**
     * @brief Apply case transformation if needed
     * @param str String to transform
     * @param toLower Transform to lowercase if true, uppercase if false
     * @return Transformed string
     */
    static std::string Transform(const std::string& str, bool toLower = true);

    /**
     * @brief Highlight matched characters in text
     * @param text The text to highlight
     * @param pattern The search pattern
     * @param caseSensitive Whether to be case-sensitive
     * @return Text with matched characters marked (e.g., "*m*a*t*c*h")
     */
    static std::string Highlight(const std::string& text, const std::string& pattern, bool caseSensitive = false);

private:
    /**
     * @brief Internal scoring algorithm using dynamic programming
     */
    static float CalculateScore(const std::string& text, const std::string& pattern, 
                               std::vector<int>& positions, const Config& config);

    /**
     * @brief Returns true if character at index i is word boundary start
     */
    static bool IsWordBoundary(const std::string& text, size_t index);

    /**
     * @brief Check if character matches (case-sensitive or not)
     */
    static bool CharMatches(char textChar, char patternChar, bool caseSensitive);
};
