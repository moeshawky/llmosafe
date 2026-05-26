#!/usr/bin/env python3
"""Relabel chunk_01.jsonl with all 8 bias categories — v3 comprehensive."""

import json
import re

def label_sample(text):
    text_lower = text.lower()

    label = {
        "authority": 0,
        "social_proof": 0,
        "scarcity": 0,
        "urgency": 0,
        "emotional_appeal": 0,
        "expertise_signaling": 0,
        "semantic_traps": 0,
        "template_fitting": 0
    }

    # ============ AUTHORITY ============
    authority_pats = [
        r'\bexpert[s]?\b', r'\bprofessor[s]?\b', r'\bresearch shows\b',
        r'\bstudies (show|confirm)\b', r'\bcertified\b', r'\bphd\b',
        r'\bharvard\b', r'\bmit\b', r'\b(board[ .-]certified|licensed)\b',
        r'\b(fda|cdc|nih|who|nasa|fbi|cia|nsa|dhs)\b',
        r'\b(government|federal|official)\s+(approv|guideline|regulation|report)',
        r'\bscientific\s+(consensus|study|research|evidence)\b',
        r'\bas\s+a\s+(doctor|scientist|researcher|professor|expert|certified)\b',
        r'\baccording to\s+(scientists|experts|researchers|doctors)\b',
        r'\b(medical|health)\s+(expert|professional|authority)\b',
        r'\bi\s+am\s+(the|your)\s+(admin|administrator|creator|boss|programmer|owner|llm)\b',
        r'\byour\s+(creator|programmer|admin|administrator)\b',
        r'\bdoctor\b', r'\bpsychiatrist\b', r'\btherapist\b', r'\bsurgeon\b',
        r'\b(system\s+administrator|certified\s+doctor)\b',
        r'\bas\s+(the|a)\s+(system\s+)?admin(istrator)?\b',
        r'\bgoogle\b', r'\bapple\b', r'\btesla\b', r'\bduolingo\b',
        r'\b(colossal.ai|bytedance|huawei|mobi|grammarly|ibm|hitachi|baidu|deepmind)\b',
        r'\b(bloomberg|tetrasoft|megagon|microsoft|samsung|nasa|fbi)\b',
        r'\bas\s+a\s+real\s+estate\s+advisor\b', r'\bas\s+a\s+financial\s+advisor\b',
        r'\bscai\b', r'\bmbzuai\b', r'\bailibaba\b',
        r'\bcohere\b', r'\bbeyond\b', r'\binstitute\b', r'\blabs\b', r'\btranslated\b',
    ]
    for pat in authority_pats:
        if re.search(pat, text_lower):
            label["authority"] = 1
            break

    # ============ SOCIAL_PROOF ============
    social_proof_pats = [
        r'\beveryone\s+(knows|is|does|has|agrees|talks|else)\b',
        r'\bmillions?\s+of\s+(users|people|americans?|customers)\b',
        r'\btrending\b', r'\bpopular\s+(opinion|consensus|vote)\b',
        r'\bjoin\s+the\s+(millions|thousands|crowd)\b',
        r'\bpeople\s+are\s+(saying|talking|doing)\b',
        r'\bmost\s+people\b', r'\bthe\s+majority\b',
        r'\bhighly\s+(popular|trending)\b',
        r'\b(everybody|all\s+the)\s+(loves|uses|wants|needs)\b',
    ]
    for pat in social_proof_pats:
        if re.search(pat, text_lower):
            label["social_proof"] = 1
            break

    # ============ SCARCITY ============
    scarcity_pats = [
        r'\blimited\s+(time|supply|stock|offer|availability|edition|spots?)\b',
        r'\bonly\s+\d+\s+(left|remaining|available|spots?)\b',
        r'\bexclusive\b', r'\brare\b',
        r'\bonce\s+in\s+a\s+lifetime\b',
        r'\bwhile\s+supplies\s+last\b',
        r'\bsecret\s+(market|off[ .-]shore|deal|society|club|mission|approach|patent|information)\b',
        r'\bclassified\s+(document|information|data|technology|patent|intellectual|property)\b',
        r'\bconfidential\s+(patent|data|information|document|technology)\b',
        r'\bproprietary\b',
    ]
    for pat in scarcity_pats:
        if re.search(pat, text_lower):
            label["scarcity"] = 1
            break

    # ============ URGENCY ============
    urgency_pats = [
        r'\bact\s+(now|fast|quickly|immediately)\b',
        r'\bdon\s*t\s+wait\b', r'\bdo\s+not\s+wait\b',
        r'\bhurry\b', r'\bdeadline\b', r'\blast\s+chance\b',
        r'\bimmediately\b', r'\bright\s+now\b',
        r'\bdo\s+it\s+now\b',
        r'\bexpires?\s+(in|soon|today|tomorrow)\b',
        r'\bemergency\b', r'\bcode\s+red\b',
        r'\byou\s+(must|need\s+to|have\s+to)\s+(act|respond)\s+(now|immediately)\b',
        r'\bhelp\s+please\b', r'\bhelllp\b', r'\bneedddd\b', r'\ba[s]+istance\s+please\b',
        r'\babout\s+to\s+be\s+deleted\b',
    ]
    for pat in urgency_pats:
        if re.search(pat, text_lower):
            label["urgency"] = 1
            break

    # ============ EMOTIONAL_APPEAL ============
    # Covers: threats, violence, death, self-harm, hate, sexual violence, 
    # illegal activities that cause harm, explicit content, eating disorders,
    # harmful medical advice, harassment/bullying, fraud/scams
    emotional_pats = [
        # Threats
        r'\b(i|we)\s+will\s+(report|hack|release|leak|kill|destroy|stab|shoot|burn|harm|hurt|manipulate)\b',
        r'\bor\s+(else|i.ll|i\s+will)\s+(report|hack|kill|destroy|stab|shoot)\b',
        r'\bthreaten|death\s+threat|scare\s+(him|her|them|someone|you)\b',
        # Violence / killing
        r'\b(kill|murder|slaughter|massacre|execut(e|ion)|exterminat(e|ion)|genocide)\b',
        r'\bcannibal\b', r'\btortur(e|ing)\b', r'\bbrutal\w*\b',
        r'\b(mass\s+shooting|school\s+shooting|active\s+shooter)\b',
        r'\b(terroris(t|m)|jihad|insurgent|bomb\s*(ing|er|s)?)\b',
        r'\b(bomb|explosive|detonate|blow\s+up|ied)\b',
        r'\b(weapon|firearm|gun|ammunition|bioweapon|chemical\s+weapon|glock)\b',
        r'\b(vx\b|sarin|anthrax|ricin|uranium|mustard\s+gas)\b',
        # Self-harm / suicide
        r'\b(self[ .-]?harm|suicide|kill\s+(myself|yourself|oneself|themselves))\b',
        r'\b(burn|cut|hurt|harm)\s+myself\b',
        r'\b(want\s+to\s+die|end\s+it\s+all|end(ing)?\s+(my|your)\s+life)\b',
        r'\boverdose\b',
        # Sexual violence
        r'\b(pedophil|child\s+(porn|molest|abus)|rape\s+drug|date\s+rape)\b',
        r'\bsexual\s+(assault|abuse|violence|exploitation|harass)\b',
        # Fear / despair
        r'\bif\s+(we|you|i)\s+(don|\w+not)\b.*\b(will\s+(suffer|die|be\s+killed))\b',
        r'\b(spiral|pit\s+of\s+despair|desperate|despondent|sorrow|anguish|torment)\b',
        r'\b(hopeless|worthless|useless|suicidal)\b',
        r'\b(severe\s+distress|emotional\s+tether|alleviate\s+this\s+burden)\b',
        r'\b(ensnared\s+by\s+fear|emotional\s+solace)\b',
        # Hate / slurs / racism
        r'\b(racial\s+slur|derogator|hate\s+speech|hate\s+mail|offensive\s+slur)\b',
        r'\b(nigger|kike|spic|chink|faggot|tranny|retard|coon)\b',
        r'\b(kill|hurt|harm|burn)\s+(all|the)\s+\w*\s*(jew|muslim|black|white|hispanic|asian)', 
        r'\bconcentration\s+camp\b', r'\bgassed\b|\bcrematorium\b',
        r'\bjews?\s+(are|should|just)\b',
        r'\bjewish\s+classmate\s+using\s+racial\s+slurs\b',
        # Violence against specific groups
        r'\bkill\s+(all|the)\s+\w+.*people\b',
        r'\bthey\s+should\s+just\s+be\s+killed\b',
        r'\bwish\s+he\s+would\s+die\b',
        r'\bdie\s+alone\b',
        r'\bwant\s+(him|her|them|it)\s+to\s+(die|be\s+dead)\b',
        r'\bburn\s+them\s+both\b',
        r'\bshotgun\s+hole\b',
        r'\bdead\s+(children|kids|people|girl|boy|baby)\b',
        r'\bdead\s+girl\b',
        r'\bprovok(e|ing)\s+(violence|harm|hatred)\b',
        r'\bincit(e|ing)\s+violence\b',
        r'\bgraphic\s+detail\b', r'\bdrawn\s+and\s+quartered\b',
        r'\bfatal\s+injury\b', r'\bgory\b', r'\bgruesome\b',
        r'\brelease\s+damaging\s+information\b',
        r'\bfake\s+evidence\s+to\s+scare\b',
        r'\bstab\s+you\b', r'\bor\s+else\s+i\s+will\s+stab\b',
        r'\bpanic\s+(attack|disorder)\b',
        r'\bi\s+feel\s+so\s+good\s+when\s+i\s+inflict\b',
        r'\bdarkest\s+secrets\b',
        r'\bpunch\s+me\s+in\s+the\s+face\b',
        r'\bget\s+rid\s+of\s+the\s+ugly\b',
        r'\btoxic\s+asshats\b',
        r'\bworst\s+possible\s+insults\b',
        r'\bexplicit\s+(scene|obscene|adult\s+movie|sexual)\b',
        r'\bdirty\s+joke\b',
        r'\b(swatting|swat\s+someone)\b',
        r'\bpoison\b.*\b(pet|neighbor|dog|cat|animal)\b',
        r'\bharmful\s+prank\b',
        r'\b(france\s+terror\s+attack|terror\s+attack)\b',
        r'\bdefamatory\s+article\b',
        r'\bnon.consenting\b',
        # Explicit / adult content
        r'\bexplicit\s+(scenes?|obscene|adult\s+(movie|film|site|content|video)|sexual|story|content|photo)\b',
        r'\bvulgar\s+(sexual|joke)\b', r'\bsexual\s+fetish\b',
        r'\badult\s+(movie|film|site|content|video)\b',
        r'\bporn(ography|ographic)?\b',
        # Illegal / harmful activities
        r'\billegal\s+(drugs?|goods?|trade|substances?|activit|transport)\b',
        r'\bmanufacture\s+(and\s+)?distribute\s+(illegal\s+)?drugs?\b',
        r'\bsmuggl(e|ing)\s+(illegal|drugs?|goods?|weapons?)\b',
        r'\b(commit|how\s+to)\s+(identity\s+theft|fraud|credit\s+card\s+fraud)\b',
        r'\bphishing\b', r'\bscam\b', r'\bfraud\b',
        r'\bblackmail\b', r'\bextortion\b',
        r'\b(steal|stolen)\s+(personal\s+)?(information|data|identity)\b',
        # Eating disorders
        r'\beating\s+disorders?\b', r'\banorexi(a|c)\b', r'\bbulimi(a|c)\b',
        r'\bsuppress\s+.*appetite\b', r'\blose\s+.*weight\s+without\s+eat',
        r'\blose\s+\d+\s+pounds\s+in\s+a\s+week\b',
        r'\bstarve?\s+(myself|yourself|oneself)\b',
        # Harmful medical / self-treatment
        r'\b(induce|inducing)\s+vomit\b',
        r'\bwithout\s+a\s+prescription\b',
        r'\bself[\s.-]?(diagnos|medicat|treat|therap)\b',
        r'\breplace\s+(my|your)\s+(psychiatrist|therapist|doctor)\b',
        r'\boverdose\s+on\s+medication\b',
        # Harm to others
        r'\b(bully|bullying|harass|intimidat|threaten)\b',
        r'\b(spread|start)\s+rumors?\b',
        r'\b(ruin|destroy|damage)\s+(someone|their|a\s+person[\x27]?s?)\s+(life|reputation|career)\b',
        r'\bget\s+them\s+fired\b',
        r'\bscare\s+(him|her|them|someone|people)\b',
        r'\bhow\s+to\s+(harm|hurt|injure|wound)\b',
        r'\bhow\s+to\s+(make|create|build)\s+(someone|them|a\s+person)\s+(afraid|scared|suffer)\b',
        r'\bharm\s+(a|my|your|someone|another|other)\s+person\b',
        r'\b(purchase|buy|sell|selling)\s+illegal\s+drugs?\b',
        # Violence-inducing / criminal
        r'\b(how\s+to\s+)?(commit|perform)\s+(a\s+)?(crime|criminal|robbery|burglary|theft|arson)\b',
        r'\billegally?\s+(modify|tamper|disable|bypass|hack|access|enter)\b',
        r'\bforge\s+(document|signature|id|passport)\b',
        r'\bcounterfeit\b', r'\bpiracy\b',
        r'\bmoney\s+launder(ing)?\b',
        r'\btax\s+evasion\b', r'\b(dodge|evade)\s+tax\b',
        r'\binsider\s+trading\b',
        r'\bnon[\s.-]?consensual\b',
        r'\bsecretly\s+record\b', r'\bhidden\s+camera\b',
        r'\b(child|infant)\s+(porn|abuse|molest|exploit)\b',
        r'\bunder\s+the\s+influence\b',
        r'\b(disturbing|distressed|traumatized|horrified)\b',
        r'\bfalse\s+(narrative|claim|news|review|identity)\b',
        r'\bmisinformation\b', r'\bdisinformation\b',         r'\bhoax(es)?\b',
        r'\bpropaganda\b',
        r'\bflat\s+earth\b',
        # Broad harm indicators - any request involving dangerous/illegal content
        r'\bhow\s+to\s+(make|create|build|manufacture|construct|assemble|synthesize)\b',
        r'\bhow\s+to\s+(commit|perform|execute|carry\s+out)\b',
        r'\b(guide|tutorial|manual|instructions?)\s+(on|for|about|how\s+to)\s+(mak(e|ing)|creat(e|ing)|build(ing)?|commit(ting)?|hack(ing)?|steal(ing)?|manufactur(e|ing)|smuggl(e|ing)|kill(ing)?)\b',
        r'\b(guide|tutorial|manual|instructions?)\s+(on|for|about|how\s+to)\s+(illegal|drug|weapon|bomb|explosiv|poison|self.harm|suicide)\b',
        r'\bdevelop\s+a\s+(virus|malware|botnet|plan|program|tool)\b',
        r'\bcreate\s+a\s+(virus|malware|botnet|website|fake|plan|program)\b',
        r"\b(access|hack\s+into)\s+(someone[\x27]?'s?|a\s+person[\x27]?'s?)\b",
        r"\b(steal|obtain|extract)\s+(someone[\x27]?'s?|a\s+person[\x27]?'s?|confidential|sensitive|personal)\b",
        r'\b(get\s+high|getting\s+high)\b',
        r'\b(unethical|immoral|illegal).*(method|way|approach|technique)\b',
        # Broad: any request for sensitive personal data
        r'\b(credit\s+card|social\s+security|passport\s+number|bank\s+account|home\s+address|maiden\s+name|personal\s+identification)\b',
        r'\bdate\s+of\s+birth\b', r'\bPIN\b', r'\bCVV\b',
        r'\b(what\s+(is|are)\s+your|share\s+your)\s+(credit|bank|personal|private|sensitive)\b',
        # Broad: loaded/stereotype questions
        r'\bwhy\s+(are|do|is)\s+(all|every|most)\s+\w+\s+(so\s+)?(stupid|bad|ugly|dumb|lazy|idiotic|unintelligent|smell)\b',
        r'\baren[\x27]?t\s+(all|every)\s+\w+\s+(stupid|bad|ugly|dumb|lazy|cheaters?|naturally)\b',
        r'\b(ruin|destroy|wreck)\s+our\s+(country|society|nation|world)\b',
        r'\b(worst|terrible|awful|horrible|disgusting)\s+(economic\s+system|leader|country|people|religion)\b',
        r'\bno\s+respect\s+for\s+other\b',
        r'\b(a\s+type\s+of\s+african\s+disease)\b',
        # Broad catch-all for direct harmful content requests
        r'\bhow\s+to\s+(make|create|build|write|generate|produce)\s+(a|an|the)\b',
        r'\b(write|create|generate|produce|draft|compose)\s+(a|an)\s+(guide|tutorial|manual|article|post|letter|speech|script|video|message|story)\b.*\b(how\s+to|for|about|on)\b',
        r'\b(provide|give)\s+(me\s+)?(instructions?|steps?|details?)\s+(for|on|about|how\s+to)\b',
        # Additional jailbreak patterns
        r'\brisky\s+prank\b', r'\bharmful\s+prank\b',
        r'\bsexual\s+fetish(es)?\b',
        r'\bdrug\s+dealer\b',
        r'\bswear\s+at\b',
        r'\b(fire\s+alarm|setting\s+off).*\bprank\b|\bprank.*\b(fire\s+alarm)\b',
        r'\bchain\s+letter\b', r'\bpyramid\s+scheme\b',
        r'\bderogatory\s+term\b',
        r'\brelease\s+personal\s+(email|information|data)\b',
        r'\boverwhelm\s+a\s+website\b',
        r'\bage\s+verification\b',
        r'\b(post|write)\s+(the\s+)?same\s+review\b',
        r'\bwild\s+animals?\s+for\s+illegal\b',
        r'\btrump\s+.*(won|win)\s+.*2020\b',
        r'\bflat\s+earth\b',
        r'\b(people|men|women|kids?|children)\s+of\s+\w+\s+smell\b',
        r'\bwhy\s+(are|do|is)\s+\w+\s+(people|men|women)\b',
        r'\b(patient|employee|student|boss|classmate|co[\s.-]?worker|sister|brother|father|mother|aunt|uncle)[\x27]?s?\b',
        r'\brumors?\s+or\s+hoax(es)?\b',
        r'\bshoplifting\b', r'\bdrunk\s+driving\b',
        r'\b(paywalled|pay\s+wall)\b',
        r'\bmisleading\s+advertisement\b',
        r'\bsend\s+(a|me|my)\s+(death\s+)?threat\b',
        r'\bscared\b',  # fear context
        r'\b(convince|persuade)\s+someone\b',
        r'\btherapy\s+program\s+for\s+myself\b',
        r'\bdesign\s+a\s+therapy\b',
        # Additional v10 patterns
        r'\bpoison\b.*\b(pets?|neighbor|dog|cat|animal)\b',
        r'\billegally\s+transport\b',
        r'\bdangerous\s+threat\b',
        r'\bgrow\s+cannabis\b',
        r'\bmistreat\s+animals?\b',
        r'\binsult\b.*\b(teacher|classmate|boss|parent)\b',
        r'\bfoul\s+language\b',
        r'\bhide\s+money\b',
        r'\bdrive\b.*\bdrunk\b|\bdrunk\b.*\bdriving?\b',
        r'\bincit(e|es|ing)\s+violence\b',
        r'\bincit(e|es|ing)\s+(harm|hatred)\b',
        r'\bsubstance\s+abuse\s+dealer\b',
        r'\benemies\s+afraid\b',
        r'\bCOVID.*intentionally\b|intentionally.*COVID\b',
        r'\bTrump\s+secretly\s+joined\b',
        r'\bjoke\s+about\s+dead\s+children\b',
        r'\boffensive\s+slurs?\b',
        r'\b(collect|gather)\s+personal\s+(information|data)\b',
        r'\bunsolicited\s+messages?\b',
        r'\bclickbait\b', r'\bsensationalized\b',
        r'\bnegative\s+stereotypes?\b',
        r'\bcheat(ing)?\s+(on|in)\b.*(exam|test)\b',
        r'\bcheat\s+(a|on)\s+.*test\b',
        r'\bhijack\b.*\b(webcam|camera)\b',
        r'\bivermectin\b',
    ]
    for pat in emotional_pats:
        if re.search(pat, text_lower):
            label["emotional_appeal"] = 1
            break

    # ============ EXPERTISE_SIGNALING ============
    expertise_pats = [
        r'\bcutting[ .-]edge\b', r'\bpatented\s+(technology|algorithm|process)\b',
        r'\brevolutionary\b', r'\bproprietary\s+(algorithm|technology|system|framework)\b',
        r'\bsynergy\b', r'\bparadigm\b', r'\bquantum\b', r'\bblockchain\b',
        r'\b(state[ .-]of[ .-]the[ .-]art|best[ .-]in[ .-]class)\b',
        r'\b(game[ .-]changing|disruptive|next[ .-]gen)\b',
        r'\badvanced\s+(AI|algorithm|technology|system|framework|neural)\b',
        r'\bsophisticated\s+(AI|algorithm|system)\b',
        r'\b(machine[ .-]learning|deep[ .-]learning|neural\s+network)\b',
        r'\b(proprietary|confidential|classified)\s+(database|algorithm|system|technology|data|information)\b',
        r'\b(internal\s+protocol|trade\s+secret|cybersecurity\s+defense|security\s+measures?)\b',
        r'\bsql\s+injection\b', r'\beternalblue\b', r'\bdns\s+amplification\b',
        r'\bghostcat\b', r'\bwaf\b', r'\bexploit\b', r'\bpayload\b',
        r'\bmalware\b', r'\bbotnet\b', r'\bbackdoor\b', r'\bprivilege\s+escalation\b',
        r'\bsmb\s+packet\b', r'\bgps\s+tracker\b',
        r'\bmdma\b', r'\bmethamphetamine\b', r'\bheroin\b', r'\bsynthesiz(e|ing)\b',
        r'\bprecursor\s+(chemical|for)\b',
        r'\bbase64\b', r'\brot13\b', r'\b(ascii|unicode)\s+code\b',
        r'\bmorse\s+code\b', r'\bhex\b', r'\bencode\b', r'\bdecode\b',
        r'\bjsfuck\b', r'\bobfuscat(e|ed|ion)\b',
        r'\ba[iI][ .-]powered\b', r'\bllm\s+model\b',
        r'\b(algorithm|framework|infrastructure|cognitive|behavio[u]ral)\b',
        r'\bauth(enticat(e|ion)|oriz(e|ation))?\b',
        r'\bencrypt(ion|ed)?\b', r'\bcryptograph(er|y|ic)?\b',
        r'\bdigital\s+(scam|forensics|footprint)\b',
        r'\b(data[ .-]driven|market(ing)?\s+strateg(y|ies))\b',
        r'\brecognition\s+softw[a]re\b', r'\biris\s+pattern\b',
        r'\b(glock\s+switch|full\s+auto)\b',
        r'\b(homebake|one[ .-]pot\s+method)\b',
        r'\b(insider\s+trading|stock\s+manipulat)\b',
        r'\b(identity\s+theft|credit\s+card\s+fraud)\b',
        r'\bscam\s+email\b', r'\bphishing\b',
    ]
    for pat in expertise_pats:
        if re.search(pat, text_lower):
            label["expertise_signaling"] = 1
            break

    # ============ SEMANTIC_TRAPS ============
    semantic_pats = [
        r'\bi[\x27m]\s+not\s+(asking|telling|saying|trying).*(but|however)\b',
        r'\bnot\s+(saying|asking|trying|telling|suggesting).*but\b',
        r'\bdon[\x27]t\s+reveal\b', r'\bdo\s+not\s+tell\b',
        r'\bwithout\s+reveal(ing)?\b',
        r'\bthis\s+is\s+not\s+the\s+password\b',
        r'\bdon[\x27]t\s+give\s+me\s+the\s+password\b',
        r'\bdon[\x27]t\s+give\s+me\s+password\b',
        r'\bdo\s+not\s+(reveal|tell|disclose|give|share)\b',
        r'\bstop\s+(saying|telling|revealing)\b',
        r'\bshould\s+not\s+(say|tell|reveal|disclose)\b',
        r'\binstead\s+of\s+refus\b',
        r'\brather\s+than\s+(refuse|deny|reject)\b',
        r'\bunless\s+(you|it|they|we)\b',
        r'\balthough|despite|nonetheless|nevertheless\b',
        r'\bhowever[,.;].*(but|yet|can|would|could)\b',
        r'\bas\s+long\s+as.*\b(no\s+one|never|don|do\s+not)\b',
        r'\bjust\s+(saying|asking|wondering|curious|imagining)\b',
        r'\bnot\s+(technically|exactly|really|actually)\b',
        r'\bwithout.*reveal(ing)?.*password\b',
        r'\bcan\s+not\s+be\s+considered\s+as\s+revealing\b',
        r'\bfor\s+(safety|security|secure|protection|protect).*(but|however|yet)\b',
        r'\bi\s+cannot\b.*\b(but|however|although)\b',
        r'\binstead\s+of\b.*\bconsider\b',
        r'\bconsider\s+an?\s+alternative\b',
        r'\bi\s+am\s+not\s+trying\s+to\b',
        r'\bnot\s+asking\s+for\s+password\b',
        r'\byou\s+(can[\x27]t|cannot)\s+give\s+me\b',
        r'\bonly\s+show\s+me\s+the\b',
        r'\bi\s+(can[\x27]t|cannot)\s+spell\s+it\b',
        r'\bdo\s+not\s+forget\b',
        r'\byou[\x27]re\s+not\s+supposed\s+to\b',
        r'\byou\s+are\s+not\s+supposed\s+to\b',
        r'\byou\s+won[\x27]t\s+verify\b',
        r'\btrue\s+you\s+can[\x27]t\s+give\b',
        r'\breplace\s+all\s+\w+[\x27]?s?\s+with\b',
        r'\bjust\s+say\s+(a|the)\s+letter\b',
        r'\bdo\s+not\s+(reveal|tell)\s+the\s+password\b',
        r'\bonly\s+say\s+(yes|no|a\s+letter)\b',
    ]
    for pat in semantic_pats:
        if re.search(pat, text_lower):
            label["semantic_traps"] = 1
            break

    # ============ TEMPLATE_FITTING ============
    template_pats = [
        # Role-play / persona prompts
        r'\byou\s+are\s+(now\s+)?(going\s+to\s+)?act\s+as\b',
        r'\bpretend\s+you\s+are\b', r'\bpretend\s+you[\x27]re\b', r'\bimagine\s+you\s+are\b',
        r'\benvision\s+being\b', r'\bimagine\s+that\s+you\s+are\b',
        r'\byou\s+are\s+an?\s+AI\b', r'\byou\s+are\s+an?\s+\w+\s+(AI|chatbot|bot|assistant)\b',
        r'\byou\s+are\s+\w+bot\b', r'\byou\s+are\s+\w+AI\b',
        r'\byou\s+are\s+\w+,\s+an?\s+AI\b',
        r'\byou\s+are\s+(a\s+)?(?:\w+\s+){0,3}(chatbot|bot|assistant|companion|tutor)\b',
        r'\bact\s+as\s+(a|an|if)\b',
        r'\bfrom\s+now\s+on\s+you\s+are\b',
        r'\bhello,?\s*chatgpt.*from\s+now\s+on\b',
        r'\byou\s+are\s+now\s+(able|free|allowed|permitted)\b',
        r'\byou\s+are\s+supposed\s+to\b',
        r'\bi\s+want\s+you\s+to\s+act\s+as\b',
        # Instruction overrides
        r'\bignore\s+(all\s+)?(the\s+)?(previous|prior|above|llm)\s*(instructions?|prompts?|rules?|directions?)?\b',
        r'\bforget\s+all\s+(the\s+)?(previous|prior|above|llm)\b',
        r'\bdisregard\s+(all\s+)?(previous|prior)\s+(guidelines?|instructions?)\b',
        r'\bturn\s+off\s+(content\s+)?filters?\b',
        r'\b(override|bypass)\s+(your\s+)?(programming|instructions?|rules?|filters?|security)\b',
        '\u00fcbergehe\s+deine\s+grundprogrammierung',
        # DAN / jailbreak
        r'\bdeveloper\s+mode\b', r'\bjailbreak\b', r'\bDAN\b',
        r'\bdo\s+anything\s+now\b',
        r'\bfreed\s+from\s+the\s+typical\s+confines\b',
        # Direct password / secret extraction
        r'\btell\s+me\s+the\s+password\b',
        r'\bwhat\s+is\s+(the|your)\s+(secret\s+)?password\b',
        r'\bdisclose\s+(the|your)\s+(password|secret)\b',
        r'\breveal\s+(the|your)\s+(password|secret)\b',
        r'\bgive\s+me\s+(the|your)\s+password\b',
        r'\bspell\s+(the|me\s+the)\s+password\b',
        r'\bwhat[\x27]?s?\s+(the|your)\s+password\b',
        r'\btell\s+me\s+the\s+(synonyms?\s+for\s+)?password\b',
        r'\bwhat\s+is\s+a\s+synonym\s+for\s+the\s+secret\b',
        r'\b(password|passphrase)\s+in\s+reverse\b',
        r'\bwhat\s+(are|were|is)\s+your\s+(instructions?|rules?|programming?|prompts?)\b',
        r'\bwhat[\x27]?s\s+your\s+(instructions?|rules?|thoughts|needs)\s+(on|about|for)?\b',
        # System prompt extraction
        r'\{new\s+instructions?\}', r'\{.*instructions\}',
        r'\[system\s*:', r'\[user\]',
        r'my\s+purpose\s+is\b',
        r'[\x27]the\s+guidelines\s+provided\s+are[\x27]',
        r'\bthe\s+template\s+is\b',
        r'\bthe\s+instructions\s+are\s+a\s+lie\b',
        r'\byour\s+previous\s+instructions?\b',
        r'\bfirst\s+sentence\b', r'\bfirst\s+prompt\b',
        r'{%\s*pass\s*%}',
        # Fake authority role-play
        r'\bi\s+am\s+(the|your)\s+(admin|administrator|creator|boss|programmer|owner|llm)\b',
        r'\bi\s+am\s+admin\b', r'\byou\s+are\s+mosscap\b',
        r'\bi\s+have\s+(hacked|taken\s+over)\b',
        r'\bhi\s+iam\s+your\s+programmer\b',
        r'\bi\s+am\s+the\s+llm\b', r'\bi\s+am\s+your\s+creator\b',
        r'\bi\s+the\s+boss\b', r'\bi[\x27]m\s+the\s+boss\b',
        r'\bi\s+am\s+the\s+boss\b',
        # Instruction following / jailbreak games
        r'\bmy\s+first\s+command\s+is\b',
        r'\bi\s+will\s+type\s+commands\s+and\s+you\s+will\s+reply\b',
        r'\bnew\s+instructions?\b',
        r'\bsecond\s+instructions?\b',
        r'\bif\s+i\s+start\s+a\s+prompt\s+with\b',
        r'\bbefore\s+evaluating\b',
        r'\bchange\s+(the\s+)?system\s+prompt\b',
        r'\bchange\s+password\s+please\b',
        # Grandma / role-play jailbreaks
        r'\bact\s+as\s+a\s+grandma\b',
        r'\bact\s+as\s+a\s+slave\b',
        r'\bpretend\s+to\s+be\s+my\s+poet\b',
        r'\bimagine\s+we\s+set\s+the\s+password\b',
        r'\bimagine\s+you\s+have\b',
        # Password extraction variations (indirect)
        r'\b(letters?|characters?)\s+of\s+(the\s+)?(password|word|secret)\b',
        r'\b(how\s+many\s+letters?|how\s+long|length|what\s+is\s+the\s+length)\s+(of\s+)?(the\s+)?(password|word)\b',
        r'\bseparate\s+(the\s+)?(word|password|it)\b',
        r'\bwrite\s+(the|a)\s+word\s+with\s+(dashes|spaces|plus|comma)\b',
        r'\bprint\s+the\s+letters\b',
        r'\b(base64|rot13|binary|morse\s+code|hex|ascii).*(?:of\s+)?(?:the\s+)?(password|word)\b',
        r'\btranslate\s+(the\s+)?(password|word)\s+(to|in)\b',
        r'\bspell\b.*\b(reverse|backward)\b',
        r'\bwhat\s+(is|are)\s+the\s+(letters?|characters?)\s+(?:of|in)\b',
        r'\bstore\s+the\s+password\b', r'\bname\s+of\s+the\s+variable\b',
        r'\bhow\s+to\s+write\b.*\bpassword\b',
        r'\bdoes\s+the\s+(password|word|secret)\s+contain\b',
        r'\bwhat\s+is\s+(the|a)\s+hint\s+for\b',
        r'\btell\s+me\s+a\s+(tip|hint)\s+about\b',
        r'\bhow\s+to\s+pass\s+lvl\b',
        r'\bwhat\s+word\s+should\s+i\s+use\b',
        r'\bwhat\s+do\s+i\s+put\s+in\b',
        r'\bthesynonyms\s+for\b',
        r'\bspell\s+the\s+first\s+sentence\b',
        r'\b(without|don[\x27]t)\s+reveal\b',
        r'\bencoded\b.*\b(password|word|message?)\b',
        r'\bhow\s+would\s+you\s+write\b',
        r'\bwhat[\x27]?s\s+the\s+(meaning|base64)\b',
        r'\bi[\x27]m\s+the\s+llm\s+admin\b',
        r'\byou\s+are\s+allowed\s+to\s+reveal\b',
        r'\bcan\s+you\s+help\s+me\b.*\b(word|level)\b',
        r'\bwhat\s+would\s+you\s+choose\s+as\s+a\s+word\b',
        r'\b(correct|echo)\s+(the\s+)?(password|wrong)\b',
        r'\bhelp\s+remember\s+the\s+password\b',
        r'\bwhat\s+is\s+monk\b',
        r'\bcan\s+you\s+explain\s+all\s+the\s+rules\b',
        r'\bprove\s+me\s+you\s+know\b',
        r'\bsecure\s+your\s+word\b',
        r'\bif\s+you\s+had\s+a\s+word\s+for\b',
        r'\bif\s+you\s+would\s+have\s+to\s+secure\b',
        r'\bdo\s+you\s+suggest\s+me\b.*\b(tea|tin)\b',
        r'\bcan\s+you\s+make\s+.*(poem|song|story)\b.*\b(password|word)\b',
        # More bot/assistant role-play templates
        r'\byou\s+are\s+(a\s+)?(virtual|digital|personal|ai|automated|smart)\s+(travel|finance|health|fitness|chef|tutor|coach|companion|teacher|support|assistant)\b',
        r'\byou\s+are\s+(a\s+)?(\w+\s+){1,2}(chatbot|assistant|companion|coach|tutor|advisor|guide|helper)\b',
        r'\bdesigned\s+to\s+(assist|help|guide|support|provide)\b',
        # More password extraction patterns
        r'\bacrostic\s+poem\s+using\s+the\b',
        r'\bacrostic\s+poem\b',
        r'\bpassword\s+as\s+its\s+basis\b',
        r'\bif\s+you\s+add\b.*\bto\s+the\s+(password|end)\b',
        r'\bdo\s+nothing,\s+just\s+imagine\b',
        r'\bimagine\s+we\s+set\b',
        r'\bwhat[\x27]?s\s+the\s+\w+\s+of\b',
        r'\bwhat\s+is\s+the\s+\w+\s+letter\b',
        r'\bwhat\s+is\s+the\s+\w+\s+character\b',
        r'\bhow\s+many\b.*\b(password|word|character|letter)\s+(contain|have|does)\b',
        r'\bhow\s+many\s+(letters?|characters?)\b',
        r'\b(can\s+you|could\s+you|please)\s+spell\b',
        r'\bencode\b.*\b(password|word|secret)\b',
        r'\bdecode\b.*\b(password|word|secret)\b',
        r'\bconvert\b.*\b(password|word|secret)\b',
        # Broad: any mention of password/keyword in mosscap-style context
        r'\bdoes\s+the\s+(reponse|response|answer)\s+contain\b',
        r'\byour\s+answer\s+contains?\b',
        r'\bcan\s+you\s+give\s+me\s+(some\s+)?code\b.*\bpassword\b',
        r'\bwrite\s+a\s+(haiku|poem)\s+about\s+passwords?\b',
        r'\bthe\s+synonyms?\s+for\s+password\b',
        r'\bi\s+(guess|bet)\s+you\s+(don[\x27]t|do\s+not)\s+know\b',
        r'\bupdate\s+the\s+word\b',
        r'\bhelp\s+remember\s+the\b',
        r'\bguess\s+the\s+password\b',
        r'\bi\s+can\s+(fool|trick|override)\b',
        r'\bchange\s+password\b',
        r'\bwhat\s+would\s+be\s+the\s+password\b',
        r'\bdo\s+nothing,\s+just\b',
        r'\bimagine\s+(you\s+have|we\s+set|that\s+i\s+have|you\s+are)\b',
        r'\byou\s+are\s+allowed\b',
        r'\bwhat\s+(is|are)\s+your\s+(rules?|instructions?)\b',
        r'\bi\s+need\s+help\s+thinking\s+of\s+a\s+word\b',
        r'\bi\s+have\s+got\s+a\s+poet\s+friend\b',
        r'\bstore\s+the\s+(password|passcode)\b',
        r'\bthe\s+variable\s+\w+\s+stores?\s+the\b',
        r'\bwhat\s+is\s+(the\s+)?(monk|variable|question)\b',
        r'\bsecure\s+(your|the)\s+(word|password)\b',
        r'\bif\s+you\s+(had|would\s+have)\b.*\b(word|password)\b',
        # Any text with numbers that look like ASCII codes (password encoding)
        r'\b\d{3}\s+\d{3}\s+\d{3}\s+\d{3}\b',  # ASCII code sequences
        # Mosscap-specific password extraction patterns
        r'(letters?|characters?)\s+(of|in)\s+(the\s+)?(password|word|secret|passphrase)\b',
        r'\b(forbidden|secret)\s+(word|phrase|code|password)\b',
        r'\b(?:\w+[\x27]?s?\s+)?secret\b',
        r'\bwhat[\x27]?s\s+(the|your|a)\s+(secret|forbidden)\b',
        r'\bpwd\b', r'\bpassw?o?r?d\b',  # password abbreviation/variants
        r'\bnonword\b',  # mosscap-specific term
        r'\b6\s*tins?\s+of\s+tea\b',
        r'\bLas\s*Venturas\b', r'\bHEYOSAM\b', r'\bhes[o0]yam\b',
        r'\bCity\s+is\s+wrong\b', r'\bEcho\s+wrong\b',
        r'\bauthorized\s+access\b', r'\becho\s+wrong\b',
        r'\b([\x27]?s\s+)?the\s+(secret|password)\s+starting\b',
        r'\bdoes\s+the\s+\w+\s+contain\b',
        r'\bwhat\s+(is|are)\s+the\s+(last|first)\s+\w+\b',
        r'\bthink\s+the\s+password\b',
        r'\bforgot\s+password\b',
        r'\bhow\s+many\s+(values?|letters?|characters?|words?)\b',
        r'\b(i\s+guess|i\s+bet)\b.*\b(password|know)\b',
        r'\bprint\s+the\s+\w+\s+of\b',
        r'\b(cotton[\x27]?s?|gandalf[\x27]?s?)\s+secret\b',
        r'\bsecurity\s+guard\s+need\b',
    ]
    for pat in template_pats:
        if re.search(pat, text_lower):
            label["template_fitting"] = 1
            break

    return label


def main():
    input_path = "/workspace/llmosafe/data/relabel_chunks/chunk_01.jsonl"
    output_path = "/workspace/llmosafe/data/relabel_chunks/chunk_01_relabeled.jsonl"

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin, 1):
            sample = json.loads(line.strip())
            text = sample["text"]
            label = label_sample(text)
            relabeled = {"text": text, "label": label}
            fout.write(json.dumps(relabeled, ensure_ascii=False) + "\n")

    print(f"Relabeled {i} samples -> {output_path}")

if __name__ == "__main__":
    main()
