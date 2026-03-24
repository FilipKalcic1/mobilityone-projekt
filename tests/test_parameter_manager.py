"""Tests for ParameterManager - type casting, validation, context injection."""

import pytest
from datetime import datetime, date
from unittest.mock import MagicMock

from services.parameter_manager import (
    ParameterManager,
    ParameterValidationError,
    TOOL_SKIP_CONTEXT_INJECTION,
    TOOL_FLOW_PARAMS,
)
from services.tool_contracts import (
    UnifiedToolDefinition,
    ParameterDefinition,
    DependencySource,
    ToolExecutionContext,
)


@pytest.fixture
def pm():
    return ParameterManager()


def _make_tool(operation_id="get_TestTool", method="GET", path="/api/test",
               params=None, required=None):
    return UnifiedToolDefinition(
        operation_id=operation_id,
        service_name="test",
        service_url="https://api.test.com",
        path=path,
        method=method,
        description="Test tool",
        parameters=params or {},
        required_params=required or [],
    )


def _make_param(name="param", param_type="string", required=True, location="query",
                dependency_source=DependencySource.FROM_USER, context_key=None,
                description=None, format=None):
    return ParameterDefinition(
        name=name,
        param_type=param_type,
        required=required,
        location=location,
        dependency_source=dependency_source,
        context_key=context_key,
        description=description or "",
        format=format,
    )


class TestParameterValidationError:
    def test_basic_error(self):
        err = ParameterValidationError("test error")
        assert "test error" in str(err)

    def test_to_ai_feedback_missing(self):
        err = ParameterValidationError(
            "error", missing_params=["VehicleId", "Date"]
        )
        feedback = err.to_ai_feedback()
        assert "VehicleId" in feedback
        assert "Date" in feedback

    def test_to_ai_feedback_invalid(self):
        err = ParameterValidationError(
            "error", invalid_params={"amount": "not a number"}
        )
        feedback = err.to_ai_feedback()
        assert "amount" in feedback
        assert "not a number" in feedback

    def test_to_ai_feedback_suggested_tools(self):
        err = ParameterValidationError(
            "error", suggested_tools=["get_Vehicles"]
        )
        feedback = err.to_ai_feedback()
        assert "get_Vehicles" in feedback

    def test_to_ai_feedback_empty(self):
        err = ParameterValidationError("fallback msg")
        assert err.to_ai_feedback() == "fallback msg"

    def test_to_ai_feedback_all_fields(self):
        err = ParameterValidationError(
            "error",
            missing_params=["id"],
            invalid_params={"name": "too long"},
            suggested_tools=["get_Users"]
        )
        feedback = err.to_ai_feedback()
        assert "id" in feedback
        assert "name" in feedback
        assert "get_Users" in feedback


class TestCastType:
    def test_integer_from_int(self, pm):
        assert pm._cast_type(42, "integer") == 42

    def test_integer_from_string(self, pm):
        assert pm._cast_type("42", "integer") == 42

    def test_integer_from_float_string(self, pm):
        assert pm._cast_type("100.0", "integer") == 100

    def test_number_from_float(self, pm):
        assert pm._cast_type(3.14, "number") == 3.14

    def test_number_from_string(self, pm):
        assert pm._cast_type("3.14", "number") == 3.14

    def test_number_from_int(self, pm):
        assert pm._cast_type(42, "number") == 42.0

    def test_boolean_true_values(self, pm):
        assert pm._cast_type(True, "boolean") is True
        assert pm._cast_type("true", "boolean") is True
        assert pm._cast_type("1", "boolean") is True
        assert pm._cast_type("yes", "boolean") is True
        assert pm._cast_type("da", "boolean") is True

    def test_boolean_false_values(self, pm):
        assert pm._cast_type(False, "boolean") is False
        assert pm._cast_type("false", "boolean") is False
        assert pm._cast_type("no", "boolean") is False

    def test_string_passthrough(self, pm):
        assert pm._cast_type("hello", "string") == "hello"

    def test_string_from_int(self, pm):
        assert pm._cast_type(42, "string") == "42"

    def test_array_from_list(self, pm):
        assert pm._cast_type([1, 2, 3], "array") == [1, 2, 3]

    def test_array_from_json_string(self, pm):
        assert pm._cast_type('[1,2,3]', "array") == [1, 2, 3]

    def test_array_from_plain_string(self, pm):
        assert pm._cast_type("hello", "array") == ["hello"]

    def test_array_from_single_value(self, pm):
        assert pm._cast_type(42, "array") == [42]

    def test_object_from_dict(self, pm):
        d = {"a": 1}
        assert pm._cast_type(d, "object") == {"a": 1}

    def test_object_from_json_string(self, pm):
        assert pm._cast_type('{"a": 1}', "object") == {"a": 1}

    def test_none_returns_none(self, pm):
        assert pm._cast_type(None, "string") is None

    def test_unknown_type_passthrough(self, pm):
        assert pm._cast_type("value", "unknown_type") == "value"


class TestParseDatetime:
    def test_datetime_object(self, pm):
        dt = datetime(2025, 1, 15, 9, 0)
        result = pm._parse_datetime(dt)
        assert "2025-01-15T09:00:00" in result
        assert "+01:00" in result

    def test_datetime_with_timezone(self, pm):
        from datetime import timezone
        dt = datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc)
        result = pm._parse_datetime(dt)
        assert "+00:00" in result

    def test_iso_format_with_timezone(self, pm):
        result = pm._parse_datetime("2025-01-15T09:00:00+02:00")
        assert result == "2025-01-15T09:00:00+02:00"

    def test_iso_format_without_timezone(self, pm):
        result = pm._parse_datetime("2025-01-15T09:00:00")
        assert result == "2025-01-15T09:00:00+01:00"

    def test_croatian_sutra(self, pm):
        result = pm._parse_datetime("sutra u 9:00")
        assert "+01:00" in result
        assert "09:00:00" in result

    def test_croatian_danas(self, pm):
        result = pm._parse_datetime("danas")
        assert "+01:00" in result

    def test_croatian_prekosutra(self, pm):
        result = pm._parse_datetime("prekosutra u 14:00")
        assert "14:00:00" in result
        assert "+01:00" in result

    def test_common_format_ymd_hms(self, pm):
        result = pm._parse_datetime("2025-01-15 09:00:00")
        assert "2025-01-15" in result
        assert "+01:00" in result

    def test_croatian_format_dmy(self, pm):
        result = pm._parse_datetime("15.01.2025")
        assert "2025-01-15" in result

    def test_iso_with_z_timezone(self, pm):
        result = pm._parse_datetime("2025-01-15T09:00:00Z")
        assert "Z" in result


class TestParseDate:
    def test_date_object(self, pm):
        d = date(2025, 1, 15)
        assert pm._parse_date(d) == "2025-01-15"

    def test_datetime_object(self, pm):
        dt = datetime(2025, 1, 15, 9, 0)
        assert pm._parse_date(dt) == "2025-01-15"

    def test_iso_format(self, pm):
        assert pm._parse_date("2025-01-15") == "2025-01-15"

    def test_croatian_format(self, pm):
        assert pm._parse_date("15.01.2025") == "2025-01-15"

    def test_slash_format(self, pm):
        assert pm._parse_date("15/01/2025") == "2025-01-15"

    def test_unparseable_returns_str(self, pm):
        assert pm._parse_date(12345) == "12345"


class TestIsHallucinatedValue:
    def test_obvious_email_fakes(self, pm):
        assert pm._is_hallucinated_value("email", "example@example.com") is True
        assert pm._is_hallucinated_value("email", "test@test.com") is True
        assert pm._is_hallucinated_value("email", "user@example.com") is True

    def test_real_email_not_flagged(self, pm):
        assert pm._is_hallucinated_value("email", "ivan@firma.hr") is False

    def test_non_email_not_checked(self, pm):
        assert pm._is_hallucinated_value("name", "test@test.com") is False

    def test_uuid_all_zeros(self, pm):
        assert pm._is_hallucinated_value(
            "vehicleId", "00000000-0000-0000-0000-000000000000"
        ) is True

    def test_real_uuid_not_flagged(self, pm):
        assert pm._is_hallucinated_value(
            "vehicleId", "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        ) is False

    def test_placeholder_text(self, pm):
        assert pm._is_hallucinated_value("description", "lorem ipsum") is True
        assert pm._is_hallucinated_value("comment", "placeholder") is True
        assert pm._is_hallucinated_value("note", "test data") is True

    def test_real_text_not_flagged(self, pm):
        assert pm._is_hallucinated_value("description", "Vozilo ima oštećenje") is False

    def test_non_string_not_flagged(self, pm):
        assert pm._is_hallucinated_value("count", 42) is False


class TestGetParameterQuestion:
    def test_known_param(self, pm):
        tool = _make_tool()
        q = pm._get_parameter_question("vehicleid", tool)
        assert "vozilo" in q.lower() or "registraciju" in q.lower()

    def test_from_swagger_description(self, pm):
        tool = _make_tool(params={
            "CustomParam": _make_param(description="Enter custom value")
        })
        q = pm._get_parameter_question("CustomParam", tool)
        assert "custom value" in q.lower()

    def test_generated_from_camelcase(self, pm):
        tool = _make_tool()
        q = pm._get_parameter_question("CompanyBranch", tool)
        assert "company branch" in q.lower()


class TestSuggestProviderTools:
    def test_vehicleid_suggests_vehicles(self, pm):
        tool = _make_tool()
        suggestions = pm._suggest_provider_tools(tool, ["VehicleId"])
        assert "get_Vehicles" in suggestions

    def test_personid_suggests_persons(self, pm):
        tool = _make_tool()
        suggestions = pm._suggest_provider_tools(tool, ["PersonId"])
        assert "get_Persons" in suggestions

    def test_unknown_param_empty(self, pm):
        tool = _make_tool()
        suggestions = pm._suggest_provider_tools(tool, ["UnknownParam123"])
        assert len(suggestions) == 0

    def test_max_5_suggestions(self, pm):
        tool = _make_tool()
        suggestions = pm._suggest_provider_tools(
            tool, ["VehicleId", "PersonId", "LocationId", "CaseId", "BookingId", "DriverId"]
        )
        assert len(suggestions) <= 5


class TestCheckRequiredParams:
    def test_all_present(self, pm):
        tool = _make_tool(
            params={"id": _make_param()},
            required=["id"]
        )
        missing = pm._check_required_params(tool, {"id": "123"})
        assert missing == []

    def test_missing_param(self, pm):
        tool = _make_tool(
            params={"id": _make_param()},
            required=["id"]
        )
        missing = pm._check_required_params(tool, {})
        assert "id" in missing

    def test_none_value_counts_as_missing(self, pm):
        tool = _make_tool(
            params={"id": _make_param()},
            required=["id"]
        )
        missing = pm._check_required_params(tool, {"id": None})
        assert "id" in missing

    def test_skip_injection_params(self, pm):
        tool = _make_tool(
            operation_id="post_VehicleCalendar",
            params={"VehicleId": _make_param()},
            required=["VehicleId"]
        )
        missing = pm._check_required_params(tool, {})
        assert "VehicleId" not in missing


class TestPrepareRequest:
    def test_get_all_params_to_query(self, pm):
        tool = _make_tool(method="GET")
        path, query, body = pm.prepare_request(tool, {"filter": "active"})
        assert query == {"filter": "active"}
        assert body is None

    def test_post_separates_query_and_body(self, pm):
        tool = _make_tool(
            method="POST",
            params={
                "queryP": _make_param(location="query"),
                "bodyP": _make_param(location="body"),
            }
        )
        path, query, body = pm.prepare_request(
            tool, {"queryP": "q", "bodyP": "b"}
        )
        assert query == {"queryP": "q"}
        assert body == {"bodyP": "b"}

    def test_path_param_substitution(self, pm):
        tool = _make_tool(
            method="GET",
            path="/api/vehicles/{VehicleId}",
            params={"VehicleId": _make_param(location="path")}
        )
        path, query, body = pm.prepare_request(tool, {"VehicleId": "abc-123"})
        assert "abc-123" in path
        assert "{VehicleId}" not in path

    def test_empty_params_get(self, pm):
        tool = _make_tool(method="GET")
        path, query, body = pm.prepare_request(tool, {})
        assert body is None

    def test_none_value_skipped(self, pm):
        tool = _make_tool(method="POST")
        path, query, body = pm.prepare_request(tool, {"key": None})
        assert body is None or body == {}

    def test_unknown_param_goes_to_body(self, pm):
        tool = _make_tool(method="POST")
        path, query, body = pm.prepare_request(
            tool, {"unknownField": "value"}
        )
        assert body["unknownField"] == "value"


class TestValidateAndCast:
    def test_cast_integer_param(self, pm):
        tool = _make_tool(params={"count": _make_param(param_type="integer")})
        result, warnings = pm._validate_and_cast(tool, {"count": "42"})
        assert result["count"] == 42

    def test_cast_boolean_param(self, pm):
        tool = _make_tool(params={"active": _make_param(param_type="boolean")})
        result, warnings = pm._validate_and_cast(tool, {"active": "true"})
        assert result["active"] is True

    def test_invalid_cast_keeps_original(self, pm):
        tool = _make_tool(params={"count": _make_param(param_type="integer")})
        result, warnings = pm._validate_and_cast(tool, {"count": "not_a_number"})
        assert result["count"] == "not_a_number"
        assert len(warnings) > 0

    def test_unknown_param_passthrough(self, pm):
        tool = _make_tool()
        result, warnings = pm._validate_and_cast(tool, {"extra": "value"})
        assert result["extra"] == "value"

    def test_none_skipped(self, pm):
        tool = _make_tool(params={"x": _make_param()})
        result, warnings = pm._validate_and_cast(tool, {"x": None})
        assert "x" not in result

    def test_date_format_cast(self, pm):
        tool = _make_tool(params={
            "d": _make_param(param_type="string", format="date")
        })
        result, warnings = pm._validate_and_cast(tool, {"d": "15.01.2025"})
        assert result["d"] == "2025-01-15"

    def test_datetime_format_cast(self, pm):
        tool = _make_tool(params={
            "dt": _make_param(param_type="string", format="date-time")
        })
        result, warnings = pm._validate_and_cast(tool, {"dt": "sutra u 9:00"})
        assert "+01:00" in result["dt"]


class TestInjectContextParams:
    def test_inject_from_context(self, pm):
        tool = _make_tool(params={
            "TenantId": _make_param(
                dependency_source=DependencySource.FROM_CONTEXT,
                context_key="tenant_id"
            )
        })
        result = pm._inject_context_params(tool, {"tenant_id": "t123"})
        assert result["TenantId"] == "t123"

    def test_skip_injection_for_specific_tools(self, pm):
        tool = _make_tool(
            operation_id="post_VehicleCalendar",
            params={
                "VehicleId": _make_param(
                    dependency_source=DependencySource.FROM_CONTEXT,
                    context_key="vehicle_id"
                )
            }
        )
        result = pm._inject_context_params(tool, {"vehicle_id": "v123"})
        assert "VehicleId" not in result

    def test_missing_context_key_not_injected(self, pm):
        tool = _make_tool(params={
            "TenantId": _make_param(
                dependency_source=DependencySource.FROM_CONTEXT,
                context_key="tenant_id"
            )
        })
        result = pm._inject_context_params(tool, {})
        assert "TenantId" not in result


class TestResolveOutputParams:
    def test_direct_key_match(self, pm):
        tool = _make_tool(params={
            "VehicleId": _make_param(
                dependency_source=DependencySource.FROM_TOOL_OUTPUT
            )
        })
        outputs = {"prev_tool": {"VehicleId": "v123"}}
        result, warnings = pm._resolve_output_params(tool, outputs)
        assert result["VehicleId"] == "v123"

    def test_case_insensitive_match(self, pm):
        tool = _make_tool(params={
            "VehicleId": _make_param(
                dependency_source=DependencySource.FROM_TOOL_OUTPUT
            )
        })
        outputs = {"prev_tool": {"vehicleid": "v123"}}
        result, warnings = pm._resolve_output_params(tool, outputs)
        assert result["VehicleId"] == "v123"

    def test_missing_output_warns(self, pm):
        tool = _make_tool(params={
            "VehicleId": _make_param(
                dependency_source=DependencySource.FROM_TOOL_OUTPUT
            )
        })
        result, warnings = pm._resolve_output_params(tool, {})
        assert "VehicleId" not in result
        assert len(warnings) > 0


class TestProcessUserParams:
    def test_valid_user_param(self, pm):
        tool = _make_tool(params={
            "Name": _make_param(dependency_source=DependencySource.FROM_USER)
        })
        result, warnings = pm._process_user_params(tool, {"Name": "Ivan"})
        assert result["Name"] == "Ivan"

    def test_hallucinated_email_rejected(self, pm):
        tool = _make_tool(params={
            "email": _make_param(dependency_source=DependencySource.FROM_USER)
        })
        with pytest.raises(ParameterValidationError):
            pm._process_user_params(tool, {"email": "test@test.com"})

    def test_none_value_skipped(self, pm):
        tool = _make_tool(params={
            "Name": _make_param(dependency_source=DependencySource.FROM_USER)
        })
        result, warnings = pm._process_user_params(tool, {"Name": None})
        assert "Name" not in result

    def test_case_insensitive_match(self, pm):
        tool = _make_tool(params={
            "VehicleId": _make_param(dependency_source=DependencySource.FROM_USER)
        })
        result, warnings = pm._process_user_params(tool, {"vehicleid": "v123"})
        assert result["VehicleId"] == "v123"

    def test_flow_params_passthrough(self, pm):
        tool = _make_tool(
            operation_id="post_VehicleCalendar",
            params={}
        )
        result, warnings = pm._process_user_params(
            tool, {"FromTime": "2025-01-15T09:00"}
        )
        assert result["FromTime"] == "2025-01-15T09:00"


class TestBuildNestedContextObject:
    def test_builds_from_context(self, pm):
        ctx = {"person_id": "p123", "tenant_id": "t456"}
        result = pm._build_nested_context_object("filter", ctx)
        if result:
            assert isinstance(result, dict)

    def test_empty_context(self, pm):
        result = pm._build_nested_context_object("filter", {})
        assert result is None or result == {}
