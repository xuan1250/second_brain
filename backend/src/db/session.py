"""Async SQLite engine + session factory (WAL, FK enforcement) and taxonomy migrations/seeds."""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.config import get_settings
from src.db import models  # noqa: F401  (register tables)
from src.db.models import DomainLexicon, DomainRegistry, SubdomainRegistry, WorkspaceSettings

DEFAULT_DOMAINS: list[dict[str, Any]] = [
    {
        "id": "engineering",
        "slug": "engineering",
        "name": "Engineering & Technology",
        "epistemic_type": "empirical",
        "authority_priority": 1,
        "description": "Physical systems, electrical & mechanical engineering, control systems, and technical specifications.",
        "icon": "⚙️",
        "color_token": "cyan",
        "chunking_strategy": "academic_equations",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "economics",
        "slug": "economics",
        "name": "Economics & Finance",
        "epistemic_type": "evaluative",
        "authority_priority": 2,
        "description": "Corporate finance, financial accounting, valuation, investments, and taxation policies.",
        "icon": "📈",
        "color_token": "emerald",
        "chunking_strategy": "financial_tables",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "law",
        "slug": "law",
        "name": "Law & Regulation",
        "epistemic_type": "normative",
        "authority_priority": 1,
        "description": "Statutes, administrative regulations, case law precedents, and commercial contracts.",
        "icon": "⚖️",
        "color_token": "amber",
        "chunking_strategy": "statutory_clauses",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
    {
        "id": "general",
        "slug": "general",
        "name": "General Knowledge",
        "epistemic_type": "general",
        "authority_priority": 5,
        "description": "General domain knowledge and unclassified documentation.",
        "icon": "📄",
        "color_token": "slate",
        "chunking_strategy": "standard_markdown",
        "classifier_keywords": None,
        "is_system": True,
        "is_active": True,
    },
]

DEFAULT_SUBDOMAINS: list[dict[str, Any]] = [
    # Engineering (4)
    {
        "id": "eng.electrical",
        "domain_id": "engineering",
        "slug": "eng.electrical",
        "name": "Electrical & Power Systems",
        "standard_bodies": "IEEE, IEC, NFPA 70 (NEC), NEMA",
    },
    {
        "id": "eng.control",
        "domain_id": "engineering",
        "slug": "eng.control",
        "name": "Control Systems & Automation",
        "standard_bodies": "ISA, IEC 61131, ISO 13849",
    },
    {
        "id": "eng.mechanical",
        "domain_id": "engineering",
        "slug": "eng.mechanical",
        "name": "Mechanical & Structural Engineering",
        "standard_bodies": "ASME Boiler & Pressure Vessel Code, ISO, ASTM",
    },
    {
        "id": "eng.technology",
        "domain_id": "engineering",
        "slug": "eng.technology",
        "name": "Information Technology & Distributed Systems",
        "standard_bodies": "RFCs, W3C, ISO/IEC 27001",
    },
    # Economics (4)
    {
        "id": "econ.finance",
        "domain_id": "economics",
        "slug": "econ.finance",
        "name": "Corporate Finance & Valuation",
        "standard_bodies": "Corporate Treasury, CFA Standards",
    },
    {
        "id": "econ.accounting",
        "domain_id": "economics",
        "slug": "econ.accounting",
        "name": "Financial Accounting & Reporting",
        "standard_bodies": "US GAAP (FASB), IFRS (IASB)",
    },
    {
        "id": "econ.investment",
        "domain_id": "economics",
        "slug": "econ.investment",
        "name": "Investment & Asset Management",
        "standard_bodies": "SEC Filings (10-K, 10-Q), Investment Prospectuses",
    },
    {
        "id": "econ.taxation",
        "domain_id": "economics",
        "slug": "econ.taxation",
        "name": "Taxation & Fiscal Policy",
        "standard_bodies": "Internal Revenue Code (IRC), OECD BEPS",
    },
    # Law (4)
    {
        "id": "law.statute",
        "domain_id": "law",
        "slug": "law.statute",
        "name": "Statutory Law & Legislation",
        "standard_bodies": "Federal & National Legislative Gazettes",
    },
    {
        "id": "law.regulation",
        "domain_id": "law",
        "slug": "law.regulation",
        "name": "Administrative & Regulatory Law",
        "standard_bodies": "Code of Federal Regulations (CFR), EU Directives",
    },
    {
        "id": "law.case_law",
        "domain_id": "law",
        "slug": "law.case_law",
        "name": "Case Law & Judicial Precedent",
        "standard_bodies": "Supreme Court, Appellate Reporters",
    },
    {
        "id": "law.contract",
        "domain_id": "law",
        "slug": "law.contract",
        "name": "Contract Law & Commercial Agreements",
        "standard_bodies": "EPC Contracts, PPAs, Bilateral Instruments",
    },
    # General (1)
    {
        "id": "general.general",
        "domain_id": "general",
        "slug": "general.general",
        "name": "General Knowledge & Catch-All",
        "standard_bodies": None,
    },
]

STARTER_LEXICON: list[dict[str, str]] = [
    # 1. liability
    {
        "term": "liability",
        "domain_id": "engineering",
        "specific_sense": "Operational risk score or physical component susceptibility to failure (e.g. thermal failure liability in FMEA).",
        "contrastive_warning": "Do not confuse with financial liabilities (balance sheet debts) or legal tort liability.",
    },
    {
        "term": "liability",
        "domain_id": "economics",
        "specific_sense": "Financial debts or obligations owed by an enterprise to external creditors recorded on the balance sheet.",
        "contrastive_warning": "Do not confuse with physical component failure susceptibility or statutory legal liability.",
    },
    {
        "term": "liability",
        "domain_id": "law",
        "specific_sense": "Legal responsibility or obligation arising from civil torts, breach of contract, or statute, enforceable by court-ordered damages.",
        "contrastive_warning": "Do not confuse with accounting balance sheet liabilities or engineering component failure risk.",
    },
    # 2. depreciation
    {
        "term": "depreciation",
        "domain_id": "engineering",
        "specific_sense": "Physical wear and tear, material degradation, loss of dielectric strength in transformer oil, or fatigue wear over operating life.",
        "contrastive_warning": "Do not confuse with accounting tax write-off schedules or legal condemnation damages.",
    },
    {
        "term": "depreciation",
        "domain_id": "economics",
        "specific_sense": "Systematic allocation of the depreciable cost of an asset over its estimated useful economic life (MACRS, straight-line).",
        "contrastive_warning": "Do not confuse with physical equipment material fatigue or legal measure of damages in tort.",
    },
    {
        "term": "depreciation",
        "domain_id": "law",
        "specific_sense": "Diminution of fair market value claimed as a measure of damages in property damage torts or condemnation proceedings.",
        "contrastive_warning": "Do not confuse with accounting depreciation schedules (ASC 360) or physical mechanical wear.",
    },
    # 3. current
    {
        "term": "current",
        "domain_id": "engineering",
        "specific_sense": "Rate of electric charge flow through a conductor measured in Amperes (I = V/R); alternating (AC) or direct (DC).",
        "contrastive_warning": "Do not confuse with short-term financial assets/liabilities or prevailing legal statutes.",
    },
    {
        "term": "current",
        "domain_id": "economics",
        "specific_sense": "Short-term assets or liabilities expected to be converted or settled within one operating cycle or one fiscal year (e.g. current ratio).",
        "contrastive_warning": "Do not confuse with electric charge flow in Amperes or prevailing statutory validity.",
    },
    {
        "term": "current",
        "domain_id": "law",
        "specific_sense": "Valid and legally in force at the present moment; prevailing statutory version, not repealed or superseded.",
        "contrastive_warning": "Do not confuse with electrical amperage or accounting current balance sheet assets.",
    },
    # 4. control
    {
        "term": "control",
        "domain_id": "engineering",
        "specific_sense": "Modification of dynamic system state via feedback (PID loop, state-space transfer function, closed-loop actuator).",
        "contrastive_warning": "Do not confuse with corporate voting interest (>50%) or anti-trust regulatory jurisdiction.",
    },
    {
        "term": "control",
        "domain_id": "economics",
        "specific_sense": "Voting interest (>50% equity) or operational dominance requiring consolidated financial reporting under IFRS 10 / US GAAP.",
        "contrastive_warning": "Do not confuse with feedback control loops or regulatory agency oversight.",
    },
    {
        "term": "control",
        "domain_id": "law",
        "specific_sense": "Regulatory governance, anti-trust jurisdiction, or legal standard of corporate direction under statutory securities acts.",
        "contrastive_warning": "Do not confuse with engineering feedback actuators or accounting consolidation thresholds.",
    },
    # 5. standard
    {
        "term": "standard",
        "domain_id": "engineering",
        "specific_sense": "Technical reference specification detailing dimensions, tolerances, protocols, or pinouts (e.g. IEEE 802.3, IEC 60076).",
        "contrastive_warning": "Do not confuse with accounting standard frameworks (IFRS/GAAP) or legal evidentiary burden standards.",
    },
    {
        "term": "standard",
        "domain_id": "economics",
        "specific_sense": "Accounting rule framework governing financial recognition and measurement (e.g. IFRS 15, US GAAP ASC 606).",
        "contrastive_warning": "Do not confuse with engineering technical specifications or legal burden-of-proof standards.",
    },
    {
        "term": "standard",
        "domain_id": "law",
        "specific_sense": "Legal burden or criterion of proof and conduct (e.g. beyond a reasonable doubt, prudent person standard, strict scrutiny).",
        "contrastive_warning": "Do not confuse with engineering technical specs or financial accounting frameworks.",
    },
    # 6. equity
    {
        "term": "equity",
        "domain_id": "engineering",
        "specific_sense": "Equivalence in signal phases, balanced structural loading, or impedance matching in power transmission lines.",
        "contrastive_warning": "Do not confuse with shareholder net equity on balance sheets or legal remedies in equity.",
    },
    {
        "term": "equity",
        "domain_id": "economics",
        "specific_sense": "Net residual interest in the assets of an enterprise after deducting all liabilities (shareholder equity, paid-in capital).",
        "contrastive_warning": "Do not confuse with transmission phase balance or legal equitable remedies (injunctions).",
    },
    {
        "term": "equity",
        "domain_id": "law",
        "specific_sense": "Branch of jurisprudence founded on natural justice granting equitable remedies (injunctions, specific performance) vs money damages.",
        "contrastive_warning": "Do not confuse with corporate balance sheet equity or physical mechanical load balancing.",
    },
    # 7. plant
    {
        "term": "plant",
        "domain_id": "engineering",
        "specific_sense": "Physical industrial facility, machinery installation, generator power station, or chemical/water processing apparatus.",
        "contrastive_warning": "Do not confuse with balance sheet PP&E capital asset line items or environmental zoning permit subjects.",
    },
    {
        "term": "plant",
        "domain_id": "economics",
        "specific_sense": "Fixed capital asset category under Property, Plant & Equipment (PP&E); capital expenditure asset class subject to depreciation.",
        "contrastive_warning": "Do not confuse with physical machinery components or environmental zoning compliance.",
    },
    {
        "term": "plant",
        "domain_id": "law",
        "specific_sense": "Regulated industrial facility subject to zoning permits, operating licenses, emissions limits, or nuisance litigation.",
        "contrastive_warning": "Do not confuse with balance sheet capital expenditure assets or engineering physical apparatus.",
    },
    # 8. default
    {
        "term": "default",
        "domain_id": "engineering",
        "specific_sense": "Factory-preset parameter value, initial configuration state of a controller, register, or firmware setting.",
        "contrastive_warning": "Do not confuse with bond covenant financial failure or legal failure to answer court summons.",
    },
    {
        "term": "default",
        "domain_id": "economics",
        "specific_sense": "Failure of a debtor to meet contractual debt service obligations (interest or principal payment failure) triggering covenant breach.",
        "contrastive_warning": "Do not confuse with firmware factory reset values or court default judgments.",
    },
    {
        "term": "default",
        "domain_id": "law",
        "specific_sense": "Failure of a party to perform a legal procedural duty or answer a court summons, resulting in default judgment.",
        "contrastive_warning": "Do not confuse with firmware preset values or financial debt service default.",
    },
    # 9. instrument
    {
        "term": "instrument",
        "domain_id": "engineering",
        "specific_sense": "Precision measuring device, sensor, or control transducer (e.g. pressure gauge, oscilloscope, actuator sensor).",
        "contrastive_warning": "Do not confuse with financial securities (debt/equity instruments) or formal legal contracts/deeds.",
    },
    {
        "term": "instrument",
        "domain_id": "economics",
        "specific_sense": "Financial contract or negotiable asset conveying monetary claims or ownership (debt instrument, derivative, equity security).",
        "contrastive_warning": "Do not confuse with physical measuring tools or formal legal contracts/deeds.",
    },
    {
        "term": "instrument",
        "domain_id": "law",
        "specific_sense": "Formal written legal document executing or evidencing a right or obligation (deed, contract, will, indenture, statute).",
        "contrastive_warning": "Do not confuse with engineering measuring sensors or financial market securities.",
    },
    # 10. stress
    {
        "term": "stress",
        "domain_id": "engineering",
        "specific_sense": "Internal force per unit area within materials resisting mechanical deformation (tension, compression, shear, thermal stress).",
        "contrastive_warning": "Do not confuse with bank liquidity stress tests or legal emotional distress/duress claims.",
    },
    {
        "term": "stress",
        "domain_id": "economics",
        "specific_sense": "Financial stress testing, economic pressure, market liquidity strain, or adverse macro scenarios evaluated on bank capital.",
        "contrastive_warning": "Do not confuse with mechanical material shear stress or legal claims of coercion/duress.",
    },
    {
        "term": "stress",
        "domain_id": "law",
        "specific_sense": "Legal duress, coercion, undue pressure, or psychological damage claimed in tort, contract invalidation, or employment suits.",
        "contrastive_warning": "Do not confuse with mechanical material stress or financial institution capital stress tests.",
    },
    # 11. trách nhiệm (Vietnamese: liability)
    {
        "term": "trách nhiệm",
        "domain_id": "engineering",
        "specific_sense": "Trách nhiệm vận hành kỹ thuật, giới hạn chịu tải an toàn hoặc rủi ro hỏng hóc thiết bị trong phân tích FMEA.",
        "contrastive_warning": "Không nhầm lẫn với nợ phải trả trên bảng cân đối kế toán hoặc trách nhiệm pháp lý bồi thường thiệt hại.",
    },
    {
        "term": "trách nhiệm",
        "domain_id": "economics",
        "specific_sense": "Nợ phải trả và nghĩa vụ tài chính của doanh nghiệp đối với các chủ nợ ghi nhận trên bảng cân đối kế toán.",
        "contrastive_warning": "Không nhầm lẫn với rủi ro hỏng hóc thiết bị cơ điện hoặc trách nhiệm pháp lý dân sự/hình sự.",
    },
    {
        "term": "trách nhiệm",
        "domain_id": "law",
        "specific_sense": "Trách nhiệm pháp lý phát sinh do vi phạm hợp đồng, hành vi xâm hại dân sự (tort) hoặc quy phạm pháp luật bị cưỡng chế thi hành.",
        "contrastive_warning": "Không nhầm lẫn với khoản nợ tài chính kế toán hoặc giới hạn an toàn kỹ thuật của thiết bị.",
    },
    # 12. khấu hao (Vietnamese: depreciation)
    {
        "term": "khấu hao",
        "domain_id": "engineering",
        "specific_sense": "Sự hao mòn vật lý, suy giảm cơ lý tính hoặc suy giảm độ bền điện môi của vật tư thiết bị qua thời gian vận hành.",
        "contrastive_warning": "Không nhầm lẫn với phân bổ chi phí khấu hao kế toán hay mức giảm giá trị bồi thường pháp lý.",
    },
    {
        "term": "khấu hao",
        "domain_id": "economics",
        "specific_sense": "Sự phân bổ có hệ thống giá trị phải khấu hao của tài sản cố định vào chi phí sản xuất kinh doanh theo thời gian sử dụng hữu ích (VAS 03/IFRS).",
        "contrastive_warning": "Không nhầm lẫn với sự hao mòn vật lý thực tế của thiết bị hay tổn thất giá trị trong tố tụng bồi thường.",
    },
    {
        "term": "khấu hao",
        "domain_id": "law",
        "specific_sense": "Mức giảm giá trị tài sản được viện dẫn để xác định quy mô thiệt hại thực tế trong tố tụng bồi thường dân sự hoặc thu hồi tài sản.",
        "contrastive_warning": "Không nhầm lẫn với quy chuẩn phân bổ chi phí kế toán thuế hay sự hao mòn tự nhiên cơ lý.",
    },
    # 13. dòng (Vietnamese: current)
    {
        "term": "dòng",
        "domain_id": "engineering",
        "specific_sense": "Dòng điện, tốc độ dịch chuyển có hướng của các hạt mang điện qua tiết diện dây dẫn đo bằng Ampe (AC hoặc DC).",
        "contrastive_warning": "Không nhầm lẫn với dòng tiền tài chính (cash flow) hoặc văn bản pháp luật hiện hành.",
    },
    {
        "term": "dòng",
        "domain_id": "economics",
        "specific_sense": "Dòng tiền (tiền mặt thu vào/chi ra), dòng vốn luân chuyển hoặc các khoản mục tài sản/nợ ngắn hạn lưu động.",
        "contrastive_warning": "Không nhầm lẫn với dòng điện vật lý trong dây dẫn hoặc văn bản luật đang có hiệu lực.",
    },
    {
        "term": "dòng",
        "domain_id": "law",
        "specific_sense": "Văn bản hoặc điều khoản quy phạm pháp luật đang có hiệu lực thi hành tại thời điểm hiện tại (hiện hành), chưa bị thay thế.",
        "contrastive_warning": "Không nhầm lẫn với dòng điện cơ điện hoặc dòng tiền tài chính doanh nghiệp.",
    },
    # 14. kiểm soát (Vietnamese: control)
    {
        "term": "kiểm soát",
        "domain_id": "engineering",
        "specific_sense": "Điều khiển trạng thái hệ thống động lực học qua vòng phản hồi tự động (vòng lặp PID, bộ chấp hành, điều khiển vòng kín).",
        "contrastive_warning": "Không nhầm lẫn với quyền sở hữu vốn chi phối doanh nghiệp hay thẩm quyền thanh tra quản lý nhà nước.",
    },
    {
        "term": "kiểm soát",
        "domain_id": "economics",
        "specific_sense": "Quyền sở hữu chi phối (>50% quyền biểu quyết) hoặc khả năng chi phối chính sách tài chính đòi hỏi lập báo cáo tài chính hợp nhất.",
        "contrastive_warning": "Không nhầm lẫn with mạch điều khiển tự động hóa hoặc quy chế quản lý hành chính nhà nước.",
    },
    {
        "term": "kiểm soát",
        "domain_id": "law",
        "specific_sense": "Thẩm quyền giám sát, cưỡng chế theo quy định chống độc quyền, tập trung kinh tế hoặc quản trị doanh nghiệp theo Luật Doanh nghiệp.",
        "contrastive_warning": "Không nhầm lẫn với bộ điều khiển vòng lặp kỹ thuật hoặc tỷ lệ sở hữu vốn kế toán.",
    },
    # 15. tiêu chuẩn (Vietnamese: standard)
    {
        "term": "tiêu chuẩn",
        "domain_id": "engineering",
        "specific_sense": "Quy chuẩn kỹ thuật quy định chi tiết kích thước, dung sai, quy cách an toàn và phương pháp thử nghiệm (IEEE, IEC, TCVN).",
        "contrastive_warning": "Không nhầm lẫn với chuẩn mực kế toán tài chính hoặc chuẩn mực chứng minh pháp lý.",
    },
    {
        "term": "tiêu chuẩn",
        "domain_id": "economics",
        "specific_sense": "Chuẩn mực kế toán và lập báo cáo tài chính (VAS, IFRS, US GAAP) quy định nguyên tắc ghi nhận và trình bày số liệu.",
        "contrastive_warning": "Không nhầm lẫn với quy chuẩn kỹ thuật thiết bị hoặc tiêu chuẩn chứng cứ tố tụng.",
    },
    {
        "term": "tiêu chuẩn",
        "domain_id": "law",
        "specific_sense": "Chuẩn mực pháp lý, căn cứ chứng minh hoặc mức độ cẩn trọng theo luật định (ví dụ: chuẩn mực thận trọng, chứng minh không còn nghi ngờ).",
        "contrastive_warning": "Không nhầm lẫn với quy chuẩn kỹ thuật thiết bị đo đạc hoặc chuẩn mực kế toán tài chính.",
    },
    # 16. công bằng (Vietnamese: equity)
    {
        "term": "công bằng",
        "domain_id": "engineering",
        "specific_sense": "Sự cân bằng đối xứng giữa các pha tín hiệu, phân bố tải cơ điện đồng đều hoặc phối hợp trở kháng trên lưới truyền tải.",
        "contrastive_warning": "Không nhầm lẫn với vốn chủ sở hữu doanh nghiệp hoặc nguyên tắc công lý trong xét xử pháp luật.",
    },
    {
        "term": "công bằng",
        "domain_id": "economics",
        "specific_sense": "Vốn chủ sở hữu, giá trị tài sản thuần còn lại của doanh nghiệp sau khi khấu trừ toàn bộ nợ phải trả.",
        "contrastive_warning": "Không nhầm lẫn với cân bằng pha kỹ thuật điện hoặc nguyên tắc lẽ công bằng trong tố tụng.",
    },
    {
        "term": "công bằng",
        "domain_id": "law",
        "specific_sense": "Lẽ công bằng và công lý tự nhiên, nguyên tắc xét xử và các biện pháp chế tài công bằng (equity jurisprudence) bên cạnh luật thành văn.",
        "contrastive_warning": "Không nhầm lẫn với vốn chủ sở hữu kế toán hoặc cân bằng tải trọng cơ khí.",
    },
    # 17. nhà máy (Vietnamese: plant)
    {
        "term": "nhà máy",
        "domain_id": "engineering",
        "specific_sense": "Cơ sở vật chất vật lý, tổ hợp nhà xưởng, trạm phát điện, dây chuyền máy móc và thiết bị công nghệ xử lý.",
        "contrastive_warning": "Không nhầm lẫn với nhóm tài sản cố định hữu hình PP&E kế toán hoặc chủ thể cấp phép môi trường.",
    },
    {
        "term": "nhà máy",
        "domain_id": "economics",
        "specific_sense": "Tài sản cố định hữu hình thuộc khoản mục Nhà xưởng, Máy móc & Thiết bị (PP&E); danh mục vốn đầu tư tài sản dài hạn (CapEx).",
        "contrastive_warning": "Không nhầm lẫn với tổ hợp cơ điện vật lý hay chủ thể trách nhiệm hành chính môi trường.",
    },
    {
        "term": "nhà máy",
        "domain_id": "law",
        "specific_sense": "Cơ sở sản xuất công nghiệp là đối tượng áp dụng giấy phép môi trường, quy hoạch đất đai hoặc trách nhiệm bồi thường do gây ô nhiễm.",
        "contrastive_warning": "Không nhầm lẫn với khấu hao tài sản cố định kế toán hoặc cấu trúc kỹ thuật của dây chuyền máy móc.",
    },
    # 18. mặc định (Vietnamese: default)
    {
        "term": "mặc định",
        "domain_id": "engineering",
        "specific_sense": "Giá trị tham số cài đặt gốc hoặc trạng thái thiết lập ban đầu của bộ điều khiển, firmware hoặc thanh ghi thiết bị.",
        "contrastive_warning": "Không nhầm lẫn với sự vỡ nợ tài chính của doanh nghiệp hoặc phán quyết vắng mặt của tòa án.",
    },
    {
        "term": "mặc định",
        "domain_id": "economics",
        "specific_sense": "Sự vỡ nợ hoặc vi phạm nghĩa vụ thanh toán nợ gốc/lãi khi đến hạn đối với hợp đồng tín dụng hoặc trái phiếu phát hành.",
        "contrastive_warning": "Không nhầm lẫn với thiết lập cấu hình sẵn của thiết bị hoặc phán quyết xử vắng mặt của tòa án.",
    },
    {
        "term": "mặc định",
        "domain_id": "law",
        "specific_sense": "Tình trạng một bên không thực hiện nghĩa vụ tố tụng hoặc không chấp hành lệnh triệu tập, dẫn đến bản án hoặc phán quyết xử vắng mặt.",
        "contrastive_warning": "Không nhầm lẫn với tham số cài đặt mặc định của máy móc hoặc vi phạm thanh toán nợ vay.",
    },
    # 19. công cụ (Vietnamese: instrument)
    {
        "term": "công cụ",
        "domain_id": "engineering",
        "specific_sense": "Dụng cụ đo lường chuyên dụng, cảm biến kiểm tra hoặc thiết bị hiển thị thông số cơ điện (đồng hồ áp suất, dao động ký).",
        "contrastive_warning": "Không nhầm lẫn với công cụ tài chính (chứng khoán phái sinh) hoặc văn bản pháp lý chính thức.",
    },
    {
        "term": "công cụ",
        "domain_id": "economics",
        "specific_sense": "Công cụ tài chính (chứng khoán vốn, nợ, công cụ phái sinh) xác lập quyền tài sản và nghĩa vụ tiền tệ.",
        "contrastive_warning": "Không nhầm lẫn với thiết bị đo kiểm cơ điện hoặc văn bản tố tụng pháp lý.",
    },
    {
        "term": "công cụ",
        "domain_id": "law",
        "specific_sense": "Văn bản pháp lý chính thức xác lập, chứng minh hoặc chuyển giao quyền và nghĩa vụ pháp luật (hợp đồng, di chúc, văn tự).",
        "contrastive_warning": "Không nhầm lẫn với dụng cụ cơ khí kỹ thuật hoặc chứng khoán tài chính.",
    },
    # 20. ứng suất (Vietnamese: stress)
    {
        "term": "ứng suất",
        "domain_id": "engineering",
        "specific_sense": "Nội lực sinh ra trên một đơn vị diện tích vật liệu để chống lại sự biến dạng dưới tác dụng của tải trọng ngoại lực (kéo, nén, cắt).",
        "contrastive_warning": "Không nhầm lẫn với kịch bản kiểm tra sức chịu tải tài chính hoặc tình trạng bị ép buộc/căng thẳng trong tranh tụng.",
    },
    {
        "term": "ứng suất",
        "domain_id": "economics",
        "specific_sense": "Kiểm tra sức chịu đựng tài chính (stress testing), áp lực thanh khoản hoặc biến động thị trường cực đoan lên vốn an toàn của ngân hàng.",
        "contrastive_warning": "Không nhầm lẫn với nội lực cơ học trong vật liệu hoặc tình trạng bị cưỡng ép pháp lý.",
    },
    {
        "term": "ứng suất",
        "domain_id": "law",
        "specific_sense": "Tình trạng bị cưỡng bức, áp lực tinh thần hoặc thiệt hại phi vật chất được xem xét trong tranh tụng dân sự hoặc lao động.",
        "contrastive_warning": "Không nhầm lẫn với ứng suất biến dạng cơ học vật liệu hoặc kiểm tra an toàn vốn ngân hàng.",
    },
]


def create_engine(url: str | None = None) -> AsyncEngine:
    url = url or get_settings().sqlite_url
    if url.startswith("sqlite") and ":memory:" not in url:
        db_path = url.split(":///", 1)[1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(url, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_conn, _record):
        journal_mode = os.environ.get("SQLITE_JOURNAL_MODE", "WAL").strip().upper()
        cur = dbapi_conn.cursor()
        cur.execute(f"PRAGMA journal_mode={journal_mode}")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables, run column migrations, seed taxonomy/lexicon, and backfill legacy rows."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        def _migrate_columns(sync_conn) -> None:
            # 1. Inspect documents table
            doc_cols_info = sync_conn.execute(text("PRAGMA table_info(documents)")).fetchall()
            if doc_cols_info:
                existing_doc_cols = {col[1] for col in doc_cols_info}
                doc_migrations = [
                    ("parsing_engine", "ALTER TABLE documents ADD COLUMN parsing_engine VARCHAR"),
                    ("engine_fallback_reason", "ALTER TABLE documents ADD COLUMN engine_fallback_reason VARCHAR"),
                    ("primary_domain_id", "ALTER TABLE documents ADD COLUMN primary_domain_id VARCHAR"),
                    ("primary_subdomain_id", "ALTER TABLE documents ADD COLUMN primary_subdomain_id VARCHAR"),
                    ("jurisdiction", "ALTER TABLE documents ADD COLUMN jurisdiction VARCHAR"),
                    ("authority_tier", "ALTER TABLE documents ADD COLUMN authority_tier INTEGER DEFAULT 3"),
                    ("effective_year", "ALTER TABLE documents ADD COLUMN effective_year INTEGER"),
                    ("is_current", "ALTER TABLE documents ADD COLUMN is_current BOOLEAN DEFAULT 1"),
                    ("superseded_by_doc_id", "ALTER TABLE documents ADD COLUMN superseded_by_doc_id VARCHAR"),
                    ("classification_result", "ALTER TABLE documents ADD COLUMN classification_result TEXT"),
                ]
                for col_name, ddl in doc_migrations:
                    if col_name not in existing_doc_cols:
                        sync_conn.execute(text(ddl))

            # 2. Inspect document_chunks table
            chunk_cols_info = sync_conn.execute(text("PRAGMA table_info(document_chunks)")).fetchall()
            if chunk_cols_info:
                existing_chunk_cols = {col[1] for col in chunk_cols_info}
                chunk_migrations = [
                    ("domain_key", "ALTER TABLE document_chunks ADD COLUMN domain_key VARCHAR"),
                    ("subdomain_key", "ALTER TABLE document_chunks ADD COLUMN subdomain_key VARCHAR"),
                    ("epistemic_modality", "ALTER TABLE document_chunks ADD COLUMN epistemic_modality VARCHAR"),
                    ("contextual_prefix", "ALTER TABLE document_chunks ADD COLUMN contextual_prefix TEXT"),
                ]
                for col_name, ddl in chunk_migrations:
                    if col_name not in existing_chunk_cols:
                        sync_conn.execute(text(ddl))

            # 3. Inspect domain_registry table
            dom_cols_info = sync_conn.execute(text("PRAGMA table_info(domain_registry)")).fetchall()
            if dom_cols_info:
                existing_dom_cols = {col[1] for col in dom_cols_info}
                dom_migrations = [
                    ("icon", "ALTER TABLE domain_registry ADD COLUMN icon VARCHAR DEFAULT '📁'"),
                    ("color_token", "ALTER TABLE domain_registry ADD COLUMN color_token VARCHAR DEFAULT 'indigo'"),
                    ("chunking_strategy", "ALTER TABLE domain_registry ADD COLUMN chunking_strategy VARCHAR DEFAULT 'standard_markdown'"),
                    ("classifier_keywords", "ALTER TABLE domain_registry ADD COLUMN classifier_keywords TEXT"),
                    ("is_system", "ALTER TABLE domain_registry ADD COLUMN is_system BOOLEAN DEFAULT 0"),
                    ("is_active", "ALTER TABLE domain_registry ADD COLUMN is_active BOOLEAN DEFAULT 1"),
                ]
                for col_name, ddl in dom_migrations:
                    if col_name not in existing_dom_cols:
                        sync_conn.execute(text(ddl))

            # 4. Inspect workspace_settings table
            ws_cols_info = sync_conn.execute(text("PRAGMA table_info(workspace_settings)")).fetchall()
            if ws_cols_info:
                existing_ws_cols = {col[1] for col in ws_cols_info}
                ws_migrations = [
                    ("llm_api_keys", "ALTER TABLE workspace_settings ADD COLUMN llm_api_keys TEXT"),
                    ("llm_key_validations", "ALTER TABLE workspace_settings ADD COLUMN llm_key_validations TEXT"),
                    ("llm_compat_base_url", "ALTER TABLE workspace_settings ADD COLUMN llm_compat_base_url VARCHAR"),
                ]
                for col_name, ddl in ws_migrations:
                    if col_name not in existing_ws_cols:
                        sync_conn.execute(text(ddl))

        await conn.run_sync(_migrate_columns)

    # 3. Seed taxonomy, polysemy starter lexicon, and backfill legacy records
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        async with session.begin():
            # Seed domains
            existing_domains = (await session.execute(select(DomainRegistry))).scalars().all()
            existing_domains_map = {d.slug: d for d in existing_domains}
            for d in DEFAULT_DOMAINS:
                if d["slug"] not in existing_domains_map:
                    session.add(
                        DomainRegistry(
                            id=d["id"],
                            slug=d["slug"],
                            name=d["name"],
                            epistemic_type=d["epistemic_type"],
                            authority_priority=d["authority_priority"],
                            description=d["description"],
                            icon=d["icon"],
                            color_token=d["color_token"],
                            chunking_strategy=d["chunking_strategy"],
                            classifier_keywords=d["classifier_keywords"],
                            is_system=d["is_system"],
                            is_active=d["is_active"],
                        )
                    )
                else:
                    row = existing_domains_map[d["slug"]]
                    row.is_system = True
                    if row.icon == "📁" or not row.icon:
                        row.icon = d["icon"]
                    if row.color_token == "indigo" or not row.color_token:
                        row.color_token = d["color_token"]
                    if row.chunking_strategy == "standard_markdown" or not row.chunking_strategy:
                        row.chunking_strategy = d["chunking_strategy"]
                    session.add(row)

            # Seed subdomains
            existing_subdomains = (await session.execute(select(SubdomainRegistry))).scalars().all()
            existing_subdomain_slugs = {s.slug for s in existing_subdomains}
            for sd in DEFAULT_SUBDOMAINS:
                if sd["slug"] not in existing_subdomain_slugs:
                    session.add(
                        SubdomainRegistry(
                            id=sd["id"],
                            domain_id=sd["domain_id"],
                            slug=sd["slug"],
                            name=sd["name"],
                            standard_bodies=sd["standard_bodies"],
                        )
                    )

            # Seed bilingual starter lexicon
            existing_lexicons = (await session.execute(select(DomainLexicon))).scalars().all()
            existing_lexicon_keys = {(lex.term.strip().lower(), lex.domain_id) for lex in existing_lexicons}
            for entry in STARTER_LEXICON:
                key = (entry["term"].strip().lower(), entry["domain_id"])
                if key not in existing_lexicon_keys:
                    session.add(
                        DomainLexicon(
                            term=entry["term"].strip(),
                            domain_id=entry["domain_id"],
                            specific_sense=entry["specific_sense"],
                            contrastive_warning=entry["contrastive_warning"],
                        )
                    )

            # Backfill existing legacy rows with NULL domain to 'general'
            await session.execute(
                text(
                    "UPDATE documents "
                    "SET primary_domain_id = 'general', primary_subdomain_id = 'general.general' "
                    "WHERE primary_domain_id IS NULL"
                )
            )
            await session.execute(
                text(
                    "UPDATE document_chunks "
                    "SET domain_key = 'general', subdomain_key = 'general.general', epistemic_modality = 'general' "
                    "WHERE domain_key IS NULL"
                )
            )

            # Slim legacy document_chunks rows by clearing duplicate text and parent_content
            await session.execute(
                text(
                    "UPDATE document_chunks "
                    "SET text = '', parent_content = '' "
                    "WHERE text != '' OR parent_content != ''"
                )
            )

            # Clean up legacy orphaned ChatThread header with 0 checkpoints in chat_history.db
            await session.execute(
                text(
                    "DELETE FROM chat_threads "
                    "WHERE thread_id = '1de1cd57414b458ebdeb8ea8305bd8ac'"
                )
            )

            # Seed WorkspaceSettings singleton if absent
            ws_res = await session.execute(select(WorkspaceSettings).where(WorkspaceSettings.id == "default"))
            ws = ws_res.scalars().first()
            if ws is None:
                ws = WorkspaceSettings(id="default")
                session.add(ws)
                await session.flush()

            # Sync persisted WorkspaceSettings values into runtime get_settings()
            runtime_settings = get_settings()
            if ws.high_confidence_threshold is not None:
                runtime_settings.ingestion.high_confidence_threshold = ws.high_confidence_threshold
            if ws.default_llm_provider:
                runtime_settings.llm.provider = ws.default_llm_provider
            if ws.default_llm_model:
                runtime_settings.llm.model_name = ws.default_llm_model

            # Sync persisted credentials & validations to runtime LLM module state
            from src.rag.llm import sync_runtime_credentials

            api_keys: dict[str, str] = {}
            if ws.llm_api_keys:
                try:
                    parsed_keys = json.loads(ws.llm_api_keys)
                    if isinstance(parsed_keys, dict):
                        api_keys = {str(k): str(v) for k, v in parsed_keys.items() if v is not None}
                except Exception:
                    pass

            validations: dict[str, dict[str, Any]] = {}
            if ws.llm_key_validations:
                try:
                    parsed_vals = json.loads(ws.llm_key_validations)
                    if isinstance(parsed_vals, dict):
                        validations = {str(k): v for k, v in parsed_vals.items() if isinstance(v, dict)}
                except Exception:
                    pass

            sync_runtime_credentials(api_keys, validations, ws.llm_compat_base_url)


_engine: AsyncEngine | None = None
_engine_url: str | None = None
_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _engine_url, _factory
    target_url = get_settings().sqlite_url
    if _engine is not None and _engine_url != target_url:
        try:
            _engine.sync_engine.dispose()
        except Exception:
            pass
        _engine = None
        _factory = None
    if _engine is None:
        _engine = create_engine(target_url)
        _engine_url = target_url
        _factory = create_session_factory(_engine)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _factory is not None
    return _factory


async def close_engine() -> None:
    """Flush WAL checkpoint to TRUNCATE, dispose SQLAlchemy engine, and reset singletons."""
    global _engine, _engine_url, _factory
    if _engine is not None:
        try:
            async with _engine.begin() as conn:
                await conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        except Exception:
            pass
        try:
            await _engine.dispose()
        except Exception:
            pass
        _engine = None
        _engine_url = None
        _factory = None


async def vacuum_sqlite_db(engine: AsyncEngine) -> None:
    """Execute VACUUM and PRAGMA wal_checkpoint(TRUNCATE) outside of any transaction with autocommit.

    Handles both AsyncEngine and sync Engine. Outside transactions prevents
    sqlite3.OperationalError: cannot VACUUM from within a transaction.
    """
    if hasattr(engine, "sync_engine"):
        async with engine.connect() as conn:
            autocommit_conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await autocommit_conn.execute(text("VACUUM"))
            await autocommit_conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
    else:
        with engine.connect() as conn:
            autocommit_conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            autocommit_conn.execute(text("VACUUM"))
            autocommit_conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session
