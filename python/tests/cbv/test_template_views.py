"""
Tests for Django-Bolt's Django-style template class-based views.

These views render Django templates and share the data, mutation, and
validation logic with the API ViewSets. The write views validate submitted
form data against the same Serializer schemas used by the API paths.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from msgspec import Meta

from django_bolt import BoltAPI
from django_bolt.serializers import Serializer
from django_bolt.testing import TestClient
from django_bolt.views import (
    CreateView,
    DeleteView,
    DetailView,
    FormView,
    ListView,
    TemplateView,
    UpdateView,
)
from tests.test_models import Article

# --- Schemas ---


class ArticleCreateSchema(Serializer):
    title: Annotated[str, Meta(min_length=1)]
    content: Annotated[str, Meta(min_length=1)]
    author: Annotated[str, Meta(min_length=1)]


class ArticleUpdateSchema(Serializer):
    title: Annotated[str, Meta(min_length=1)]
    content: Annotated[str, Meta(min_length=1)]
    author: Annotated[str, Meta(min_length=1)]
    is_published: bool


class ContactSchema(Serializer):
    name: Annotated[str, Meta(min_length=1)]
    message: Annotated[str, Meta(min_length=1)]


# --- Fixtures ---


@pytest.fixture(scope="module")
def api():
    api = BoltAPI()

    @api.view("/about")
    class AboutView(TemplateView):
        template_name = "test_dashboard.html"

        async def get(self, request):
            context = self.get_context_data(title="About Us")
            return self.render_to_response(context)

    @api.view("/articles")
    class ArticleListView(ListView):
        queryset = Article.objects.all()
        template_name = "test_article_list.html"
        context_object_name = "articles"

    @api.view("/articles/{pk}")
    class ArticleDetailView(DetailView):
        queryset = Article.objects.all()
        template_name = "test_article_detail.html"

    @api.view("/articles/new", methods=["GET", "POST"])
    class ArticleCreateView(CreateView):
        queryset = Article.objects.all()
        template_name = "test_article_form.html"
        form_serializer_class = ArticleCreateSchema
        success_url = "/articles"

    @api.view("/articles/{pk}/edit", methods=["GET", "POST"])
    class ArticleUpdateView(UpdateView):
        queryset = Article.objects.all()
        template_name = "test_article_form.html"
        form_serializer_class = ArticleUpdateSchema
        success_url = "/articles"

    @api.view("/articles/{pk}/delete", methods=["GET", "POST"])
    class ArticleDeleteView(DeleteView):
        queryset = Article.objects.all()
        template_name = "test_article_confirm_delete.html"
        success_url = "/articles"

    @api.view("/contact", methods=["GET", "POST"])
    class ContactView(FormView):
        template_name = "test_contact_form.html"
        form_serializer_class = ContactSchema
        success_url = "/contact/thanks"

    @api.view("/contact/thanks")
    class ContactThanksView(TemplateView):
        template_name = "test_thanks.html"

    return api


@pytest.fixture(scope="module")
def client(api):
    return TestClient(api)


@pytest.fixture
def sample_articles(db):
    from asgiref.sync import async_to_sync  # noqa: PLC0415

    articles = []
    for i in range(1, 4):
        articles.append(
            async_to_sync(Article.objects.acreate)(
                title=f"Article {i}",
                content=f"Content {i}",
                author="Test Author",
                is_published=(i % 2 == 0),
            )
        )
    return articles


# --- TemplateView ---


@pytest.mark.django_db(transaction=True)
class TestTemplateView:
    def test_renders_template(self, client):
        response = client.get("/about")
        assert response.status_code == 200
        assert "<h1>About Us</h1>" in response.text

    def test_html_content_type(self, client):
        response = client.get("/about")
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type


# --- ListView ---


@pytest.mark.django_db(transaction=True)
class TestListView:
    def test_renders_objects(self, client, sample_articles):
        response = client.get("/articles")
        assert response.status_code == 200
        for article in sample_articles:
            assert article.title in response.text

    def test_object_list_context(self, client, sample_articles):
        response = client.get("/articles")
        assert response.status_code == 200
        assert "Article 1" in response.text
        assert "Article 3" in response.text


# --- DetailView ---


@pytest.mark.django_db(transaction=True)
class TestDetailView:
    def test_renders_single_object(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.get(f"/articles/{pk}")
        assert response.status_code == 200
        assert "Article 1" in response.text
        assert "Content 1" in response.text

    def test_missing_object_returns_404(self, client):
        response = client.get("/articles/999999")
        assert response.status_code == 404


# --- FormView ---


@pytest.mark.django_db(transaction=True)
class TestFormView:
    def test_get_renders_form(self, client):
        response = client.get("/contact")
        assert response.status_code == 200

    def test_post_valid_redirects(self, client):
        response = client.post(
            "/contact",
            data={"name": "Ada", "message": "Hello"},
        )
        assert len(response.history) == 1
        assert response.history[0].status_code == 302
        assert response.history[0].headers.get("location") == "/contact/thanks"
        assert response.status_code == 200
        assert "<h1>Thanks</h1>" in response.text

    def test_post_invalid_rerenders_with_errors(self, client):
        response = client.post(
            "/contact",
            data={"name": "", "message": ""},
        )
        assert response.status_code == 422
        assert "error" in response.text.lower()


# --- CreateView ---


@pytest.mark.django_db(transaction=True)
class TestCreateView:
    def test_get_renders_form(self, client):
        response = client.get("/articles/new")
        assert response.status_code == 200

    def test_post_creates_and_redirects(self, client):
        response = client.post(
            "/articles/new",
            data={"title": "New Article", "content": "Body", "author": "Ada"},
        )
        assert len(response.history) == 1
        assert response.history[0].status_code == 302
        assert response.history[0].headers.get("location") == "/articles"
        assert response.status_code == 200
        assert Article.objects.filter(title="New Article").exists()

    def test_post_invalid_rerenders_with_errors(self, client):
        response = client.post(
            "/articles/new",
            data={"title": "", "content": "", "author": ""},
        )
        assert response.status_code == 422


# --- UpdateView ---


@pytest.mark.django_db(transaction=True)
class TestUpdateView:
    def test_get_renders_form(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.get(f"/articles/{pk}/edit")
        assert response.status_code == 200

    def test_post_updates_and_redirects(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.post(
            f"/articles/{pk}/edit",
            data={
                "title": "Updated Title",
                "content": "Updated Body",
                "author": "Ada",
                "is_published": "true",
            },
        )
        assert len(response.history) == 1
        assert response.history[0].status_code == 302
        assert response.history[0].headers.get("location") == "/articles"
        assert response.status_code == 200
        article = Article.objects.get(pk=pk)
        assert article.title == "Updated Title"
        assert article.is_published is True

    def test_post_invalid_rerenders_with_errors(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.post(
            f"/articles/{pk}/edit",
            data={"title": "", "content": "", "author": "", "is_published": "false"},
        )
        assert response.status_code == 422


# --- DeleteView ---


@pytest.mark.django_db(transaction=True)
class TestDeleteView:
    def test_get_renders_confirmation(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.get(f"/articles/{pk}/delete")
        assert response.status_code == 200
        assert "Delete Article 1" in response.text

    def test_post_deletes_and_redirects(self, client, sample_articles):
        pk = sample_articles[0].pk
        response = client.post(f"/articles/{pk}/delete")
        assert len(response.history) == 1
        assert response.history[0].status_code == 302
        assert response.history[0].headers.get("location") == "/articles"
        assert response.status_code == 200
        assert not Article.objects.filter(pk=pk).exists()
