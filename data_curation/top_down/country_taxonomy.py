import json

countries = {
    "Afghanistan", "Aland Islands", "Albania", "Algeria", "American Samoa", "Andorra",
    "Angola", "Anguilla", "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
    "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bonaire, Saint Eustatius and Saba",
    "Bosnia and Herzegovina", "Botswana", "Bouvet Island", "Brazil", "British Indian Ocean Territory",
    "British Virgin Islands", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China",
    "Christmas Island", "Cocos Islands", "Colombia", "Comoros", "Cook Islands", "Costa Rica", "Croatia",
    "Cuba", "Curacao", "Cyprus", "Czechia", "Democratic Republic of the Congo", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
    "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Greenland", "Grenada", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Heard Island and McDonald Islands", "Honduras", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jersey", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Montserrat", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestinian Territory", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn Islands", "Poland", "Portugal", "Puerto Rico",
    "Qatar", "Romania", "Russia", "Rwanda", "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Wallis and Futuna",
    "Western Sahara", "Yemen", "Zambia", "Zimbabwe"
}

culture_category = {
    "Cuisine": "Refers to the foods, culinary practices, and cooking methods that are unique to specific regions or cultures. This includes iconic dishes, preparation techniques, and the cultural background behind eating habits, as well as the importance of food in social and religious practices.",
    
    "Clothing": "Encompasses traditional garments, accessories, and adornments from various cultures. It includes not only clothing but also items like jewelry, headwear, and footwear that hold cultural significance, reflecting identity, status, and traditions.",
    
    "Animal & Plants": "Describes the native species, both fauna and flora, that hold cultural importance. This category includes the use of animals and plants in mythology, cuisine, traditional medicine, and environmental practices, as well as their roles in folklore and symbolism.",
    
    "Art": "Includes visual arts, sculptures, and other forms of artistic expression that represent a culture's aesthetic and artistic heritage. This encompasses paintings, sculptures, performance arts, and crafts that reflect the identity, beliefs, and historical evolution of a community.",
    
    "Architecture": "Refers to the design, style, and structures built by a particular culture. This includes traditional houses, temples, monuments, and public buildings that showcase the engineering, material use, and aesthetic values of the culture.",
    
    "Daily Life": "Covers the everyday activities, routines, and practices that define how people in a particular culture live. This includes family roles, work habits, and leisure activities, as well as practices around health, education, and community.",
    
    "Symbol": "Involves the symbols, logos, and imagery that carry cultural meaning. This category includes national flags, religious icons, mythological figures, and colors that convey beliefs, values, and identity in various contexts.",
    
    "Festival": "Encompasses cultural festivals, holidays, and ceremonies, along with the associated customs, rituals, and practices. Examples include events like Chinese New Year, Diwali, and Christmas, each rich in traditions, foods, and rituals that symbolize community and heritage."
}

def generate_country_category_pairs(output_file):
    """Generate all possible country-category pairs and save to JSONL file."""
    # Generate all possible pairs
    pairs = []
    for country in countries:
        for category in culture_category.keys():
            pair = {
                "country": country,
                "category": category,
                "metadata": {
                    "category_description": culture_category[category]
                }
            }
            pairs.append(pair)
    
    # Save to JSONL file
    with open(output_file, 'w') as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write("\n")
    
    total_pairs = len(countries) * len(culture_category)
    print(f"Generated {total_pairs} country-category pairs ({len(countries)} countries × {len(culture_category)} categories) and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Generate pairs when this file is run directly
    output_file = "country_category_pairs.jsonl"
    generate_country_category_pairs(output_file)


